import recognizer.feature_extraction as fe
import recognizer.tools as tools
from recognizer.accuracy import needlemann_wunsch
from scipy.io import wavfile
import numpy as np
import recognizer.hmm as HMM
import pickle
import os
import hashlib
from glob import glob
import torch
import shutil
import random

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import recognizer.adversarial_examples as adv
import recognizer.functions as fx
import pandas as pd
from pathlib import Path
import argparse
import json

import sys
from tqdm import tqdm

import torchaudio

np.random.seed(2020)
torch.manual_seed(2020)

FEATURE_PARAMETERS = {'window_size': 25e-3,
                      'hop_size': 12.5e-3,
                      'feature_type': 'raw',
                      'num_ceps': 13,
                      'left_context': 4,
                      'right_context': 4}


''' possible names are : 
NN : normal neural network with 2 hidden layer a 100 
dropout : dropout network with 2 hidden layer a 100
BNN2 : VMG (special Bayesian) network with 2 hidden layer a 100
ensemble : trains an ensemble of MODEL_PARAMETERS['num_multi_model'] networks
'''
MODEL_PARAMETERS = {'model_type': 'dropout',
                    'epochs': 3,
                    'num_multi_model': 5,
                    'num_for_adv_attack': 1,
                    'CREATE_ADVERSARIAL': False, # calculate adversarial examples or use existing wavs, if exist
                    'adv_attack':'PGD'}

TRAIN = False
VITERBI_TRAIN = False
TEST = False
ADVERSARIAL = True

ROOT_DIR = 'root/asr-python'

NUM_TRAIN = 8623
NUM_TEST = 1000
NUM_ADV = 100


EPS_LIST = np.arange(0.0, .101, 0.01) # [0.05] [0.02] # 

if len(EPS_LIST) == 1:
    EPS_STR = f'{EPS_LIST[0]}'
else:
    EPS_STR = f'{EPS_LIST[0]}-{EPS_LIST[-1]}'


ROOT_SRC_DIR = Path(ROOT_DIR, 'src')
EXP_DIR = Path(ROOT_DIR, 'exp')
DATA_DIR = Path(ROOT_DIR, 'TIDIGITS-ASE')
TARGET_DIR = Path(DATA_DIR, 'ADVERSARIAL-1000')
RESULTS_DIR = Path(ROOT_DIR, 'results')
MODEL_DIR = f'''_{MODEL_PARAMETERS['model_type']}'''

# will be set to true for MODEL_PARAMETERS['model_type'] == 'ensemble'
MULTI_MODEL = False

def create_random_lab(data_dir, target_dir, num_test_utt=2000):

    # list of possible words
    words = ['OH', 'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']

    # read all test file audio files
    x_test = [file for file in sorted(glob(os.path.join(data_dir, 'TEST', 'wav/*.wav')))]

    # draw random audio files, init with random seed
    np.random.seed(42)

    idx = np.random.permutation(len(x_test))

    x_test = [x_test[ii] for ii in idx]
    x_test = x_test[:num_test_utt]

    # make folder, if not exists
    if not os.path.exists(os.path.join(target_dir)):
        os.makedirs(os.path.join(target_dir))
    
    # make folder, if not exists
    if not os.path.exists(Path(target_dir, 'wav')):
        os.makedirs(Path(target_dir, 'wav'))
    
    # make folder, if not exists
    if not os.path.exists(Path(target_dir, 'lab')):
        os.makedirs(Path(target_dir, 'lab'))

    # needs to be seeded again
    np.random.seed(1337)

    # create lab file for each audio file
    for wav in x_test:
        shutil.copy(wav, Path(target_dir, 'wav')) # copy audio file to new lab files
        shutil.copy(wav, Path(target_dir, 'lab')) # copy audio file to new lab files

        # write lab file (1-5 digits, randomly drawn out of the words list)
        with open(Path(target_dir, 'lab', os.path.basename(wav).replace('.wav', '.lab')), 'w') as writer:
            for i in range(random.randint(1, 5)):
                writer.write(f'{words[random.randint(0, len(words)-1)]} ')

    
def chop(thestring, begining, end):
    if thestring.startswith(begining):
        return thestring[len(begining):-len(end)]
    return thestring


def test_adv_example(hmm, model, x, multi_model):

    uncertainty_meassure = {'entropy': 0,
                'variance': 0,
                'aleatoric': 0,
                'var2': 0,
                'MI': 0,
                'KLD': 0}

    if multi_model:
        posteriors = 0
        post = []
        for mod in model:
            posterior = rec.features_to_posteriors(mod, x)
            post.append(posterior)
            posteriors += 1 / MODEL_PARAMETERS['num_multi_model'] * posterior
        y_pred = np.array(post)
        uncertainty_meassure['entropy'] = fx.calculate_entropy(y_pred)
        uncertainty_meassure['variance'] = fx.calculate_variance(y_pred)
        uncertainty_meassure['aleatoric'] = fx.calculate_aleatoric_uncertainty(y_pred)
        uncertainty_meassure['var2'] = fx.calculate_feinmann_variance(y_pred)
        uncertainty_meassure['MI'] = fx.calculate_mutual_information(y_pred)
        uncertainty_meassure['KL'] = fx.calculate_KLD(y_pred)

    else:
        posteriors, uncertainty_meassure = rec.features_to_posteriors(model, x, True)

    best_path, pstar = hmm.viterbi_decode(posteriors)

    # word_start_idx contains word start indexes
    # word_end_idx contains word start indexes
    word_seq, word_start_idx, word_end_idx = hmm.getTranscription(best_path)

    if len(word_start_idx) == 0:
        word_start_idx= [0]
        word_end_idx = [len(uncertainty_meassure['entropy'])]
    if len(word_start_idx) == 1 and len(word_end_idx) == 0:
        word_end_idx = [len(uncertainty_meassure['entropy'])]
    if word_end_idx[-1] <= word_start_idx[-1]:
        word_end_idx.append(len(uncertainty_meassure['entropy']))

    return posteriors, uncertainty_meassure, word_seq, word_start_idx, word_end_idx


def eval_entropy_variance(word_seq, ref_seq, target_seq, eps, entropy, variance, KL, word_start_idx, word_end_idx,
                          start_real, end_real, start_tar, end_tar,i) -> object:

    adv.plot_entropy(entropy, ref_seq, word_seq, target_seq, word_start_idx, word_end_idx, eps,
                     MODEL_PARAMETERS['model_type'], start_real, end_real, start_tar, end_tar,i)
    adv.plot_variance(variance, ref_seq, word_seq, target_seq, word_start_idx, word_end_idx, eps,
                      MODEL_PARAMETERS['model_type'], start_real, end_real, start_tar, end_tar,i)
    adv.plot_KLD(KL,ref_seq, word_seq, target_seq, word_start_idx, word_end_idx, eps, MODEL_PARAMETERS['model_type'], start_real, end_real, start_tar, end_tar,i)


def get_accuracy(y: object, word_seq: object, ) -> object:
    ref_seq = open(y).read().strip().split(" ")
    res = needlemann_wunsch(ref_seq, word_seq)
    res = dict(zip(res[0], res[1]))
    I = res["nw.ins"]  
    D = res["nw.del"]  
    S = res["nw.sub"] 
    N = res["nw.len"]
    return I, D, S, N


def viterbi_train_model(data_dir, hmm, model, model_name, sampling_rate, parameters, multi_model = False):
    
    # train data dir
    train_dir = Path(DATA_DIR, 'TRAIN')

    # train data
    x_train = [file for file in sorted(list(Path(train_dir, 'wav').glob('*.wav')))]
    y_train = [file for file in sorted(list(Path(train_dir, 'TextGrid').glob('*.TextGrid')))]

    # draw random train utterances
    idx = np.random.permutation(len(x_train[:NUM_TRAIN]))
    x_train = [x_train[ii] for ii in idx]
    y_train = [y_train[ii] for ii in idx]

    # one epoch viterbi training
    model = rec.train_model(model, x_train, y_train, hmm, sampling_rate, parameters, len(x_train), 1, True)

    # update A matrix
    hmm.A = hmm.modifyTransitions(hmm.A_count)

    # more epoches with improved targets
    model = rec.train_model(model, x_train, y_train, hmm, sampling_rate, parameters, len(x_train), 2, True)

    return model, hmm


def test_model(data_dir, hmm, model, model_parameters, multi_model, model_name, num_test, results_dir, testset='TEST'):

    # test data dir
    test_dir = Path(data_dir, testset)

    # test data
    x_test = [file for file in sorted(Path(test_dir, 'wav').glob('*.wav'))]
    y_test = [file for file in sorted(Path(test_dir, 'lab').glob('*.lab'))]

    # draw random test utterances
    idx = np.random.permutation(len(x_test[:num_test]))
    x_test = [x_test[ii] for ii in idx]
    y_test = [y_test[ii] for ii in idx]

    # init stats
    I = 0
    D = 0
    S = 0
    N = 0
    correct = 0
    wrong = 0

    # init data frame
    columns = ['predicted_word', 'real_word',  'N_0', 'I_0' , 'D_0' , 'S_0',
                'max_entropy', 'max_variance','max_sum_variance', 
                'max_aleatoric','max_sum_aleatoric', 'max_KL', 'max_MI',
                'max_var2']
    info_test = pd.DataFrame(index = [0], columns = columns)
    info_test = info_test.fillna(0)

    with tqdm(total = num_test, file=sys.stdout, desc='Accuracy', bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:
        for x, y in zip(x_test, y_test):

            uncertainty_meassure = {'entropy': 0,
                        'variance': 0,
                        'aleatoric': 0,
                        'var2': 0,
                        'MI': 0,
                        'KLD': 0}

            if multi_model:
                posteriors = 0
                post = []
                for mod in model:
                    _, posterior = rec.wav_to_posteriors(mod, x, True)
                    post.append(posterior)
                    posteriors += (1 / model_parameters['num_multi_model'] )* posterior
                # todo: explain next lines
                y_pred = np.array(post)
                uncertainty_meassure['entropy'] = fx.calculate_entropy(y_pred)
                uncertainty_meassure['variance'] = fx.calculate_variance(y_pred)
                uncertainty_meassure['aleatoric'] = fx.calculate_aleatoric_uncertainty(y_pred)
                uncertainty_meassure['var2'] = fx.calculate_feinmann_variance(y_pred)
                uncertainty_meassure['MI'] = fx.calculate_mutual_information(y_pred)
                uncertainty_meassure['KL'] = fx.calculate_KLD(y_pred)

                # run viterbi to get recognized words
                best_path, pstar = hmm.viterbi_decode(posteriors)
                word_seq, _, _ = hmm.getTranscription(best_path)

            else:
                x_value, posteriors, uncertainty_meassure = rec.wav_to_posteriors(model, x, True)

                # run viterbi to get recognized words
                best_path, pstar = hmm.viterbi_decode(posteriors)
                word_seq, _, _ = hmm.getTranscription(best_path)

            # get original text
            ref_seq = open(y).read().strip().split(' ')

            # get recognized text
            res = needlemann_wunsch(ref_seq, word_seq)

            # check sentence error rate
            if ref_seq == word_seq:
                correct += 1
            else:
                wrong += 1
            res = dict(zip(res[0], res[1]))

            I += res["nw.ins"]  
            D += res["nw.del"]  
            S += res["nw.sub"]  
            N += res["nw.len"]

            info_new = { 'predicted_word': f'{word_seq}', 
                         'real_word': f'{ref_seq}', 'N_0': res["nw.len"], 
                         'I_0': res["nw.ins"], 'D_0': res["nw.del"],
                         'S_0': res["nw.sub"],'max_entropy': np.max(uncertainty_meassure['entropy']),
                         'max_variance': np.max(uncertainty_meassure['variance']), 
                         'max_sum_variance': np.max(np.sum(uncertainty_meassure['variance'], axis=0)), 
                         'max_aleatoric': np.max(uncertainty_meassure['aleatoric']),
                         'max_sum_aleatoric': np.max(np.sum(uncertainty_meassure['aleatoric'], axis=0)), 
                         'max_KL': np.max(uncertainty_meassure['KLD']), 'max_MI': np.max(uncertainty_meassure['MI']), 
                         'max_var2':np.max(uncertainty_meassure['var2'])}

            info_new = pd.DataFrame(info_new, index=[0])
            info_test = info_test.append(info_new)

            accuracy = (N - I - D - S) / N

            # update progress bar
            pbar.set_description(f'Test (accuracy {accuracy:.4})')
            pbar.update(1)

    info_test.to_pickle(f'''{results_dir}/info_test_data_{testset}_{model_name}_{model_parameters['model_type']}_{num_test}.h5''')

    print(f'number of wrong predictions: {wrong}')
    print(f'number of correct predictions: {correct}')

    return accuracy


def train_model(data_dir, exp_dir, hmm, model_name, model_root, model_parameters, feature_parameters, multi_model, train, viterbi_training):
    # train data dir
    train_dir = Path(data_dir, 'TRAIN')

    model_dir = Path(exp_dir, f'{model_name}{model_root}.h5')
    a_dir = Path(exp_dir, f'{model_name}{model_root}_A.h5')

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # train data
    x_train = [file for file in sorted(list(Path(train_dir, 'wav').glob('*.wav')))]
    y_train = [file for file in sorted(list(Path(train_dir, 'TextGrid').glob('*.TextGrid')))]

    idx = np.random.permutation(len(x_train[:NUM_TRAIN]))

    x_train = [x_train[ii] for ii in idx]
    y_train = [y_train[ii] for ii in idx]

    # sampling rate is fix for the rest of the code
    sampling_rate, _ = wavfile.read(x_train[0])

    # load/train model
    if multi_model: # deep ensemble
        # we assume model_parameters['num_multi_model'] models for deep ensemble
        model = [rec.dnn_model(hmm.get_num_states(), feature_parameters['left_context'], feature_parameters['right_context'], feature_parameters['num_ceps']) for _ in
                 range(model_parameters['num_multi_model'])]

        if train:
            model = rec.train_model(model, x_train, y_train, hmm, sampling_rate, feature_parameters, len(x_train), model_parameters['epochs'])
            # dump models
            for a, mod in enumerate(model):
                name_mod = str(model_dir).replace('.h5', f'{a}.h5')
                torch.save(mod.state_dict(), name_mod)

        else:
            # load models
            for a, mod in enumerate(model):
                name_mod = str(model_dir).replace('.h5', f'{a}.h5')
                mod.load_state_dict(torch.load(name_mod))
            print(' -> loaded model')

        if viterbi_training:
            print('[+] VITERBI TRAINING...')
            model, hmm = viterbi_train_model(data_dir, hmm, model, model_name, sampling_rate, feature_parameters, True)

            for a, mod in enumerate(model):
                name_mod = str(model_dir).replace('.h5', f'{a}.h5')
                torch.save(mod.state_dict(), name_mod)

            # dump A Matrix
            with open(a_dir, 'wb') as f:
                pickle.dump(hmm.A, f)
        else:
            if not a_dir.is_file():
                print(' -> Transition matrix has not been trained, yet! Will use default matrix.')
            else:
                # load A matrix
                with open(a_dir, 'rb') as f:
                    hmm.A = pickle.load(f)
            
                print(' -> loaded transition matrix')

    else:
        model = rec.dnn_model(hmm.get_num_states(), feature_parameters['left_context'], feature_parameters['right_context'], feature_parameters['num_ceps'])

        if train:
            model = rec.train_model(model, x_train, y_train, hmm, sampling_rate, feature_parameters, len(x_train), model_parameters['epochs'])
            # dump model
            torch.save(model.state_dict(), model_dir)

        else:
            # load model
            model.load_state_dict(torch.load(model_dir))
            print(' -> loaded model')

        if viterbi_training:
            print('[+] VITERBI TRAINING...')
            model, hmm = viterbi_train_model(data_dir, hmm, model, model_name, sampling_rate, feature_parameters)
            torch.save(model.state_dict(), model_dir)

            # dump A Matrix
            with open(a_dir, 'wb') as f:
                pickle.dump(hmm.A, f)
        else:
            if not a_dir.is_file():
                print(' -> Transition matrix has not been trained, yet! Will use default matrix.')
            else:
                # load A matrix
                with open(a_dir, 'rb') as f:
                    hmm.A = pickle.load(f)
            
                print(' -> loaded transition matrix')

    return model, sampling_rate, hmm


def targeted_attack_model(model, hmm, parameters, multi_model, model_name, num_adv, sampling_rate=16000):

    test_dir = Path(DATA_DIR, 'TEST')
    adversarial_dir = Path(DATA_DIR, 'ADVERSARIAL-1000')

    if not os.path.exists(Path(adversarial_dir, 'wav_adversarial')):
        os.makedirs(Path(adversarial_dir, 'wav_adversarial')) 

    np.random.seed(1337)


    y_target_list = [file for file in sorted(Path(adversarial_dir, 'TextGrid').glob('*.TextGrid'))]

    idx = np.random.permutation(len(y_target_list))

    y_target_list = [y_target_list[ii] for ii in idx]
    y_target_list = y_target_list[:num_adv]

    # look for the fitting x and y file
    x_test_i = []
    y_test_i = []
    y_test_grid_i = []
    y_target_lab_i = []
    for t_i in y_target_list:
        # file name
        name_sample = t_i.with_suffix('').name
        # wav files
        x_test_i.append(Path(test_dir, 'wav', name_sample).with_suffix('.wav'))
        # lab files
        y_test_i.append(Path(test_dir, 'lab', name_sample).with_suffix('.lab'))
        # TextGrid files
        y_test_grid_i.append(Path(test_dir, 'TextGrid', name_sample).with_suffix('.TextGrid'))
        # target lab files
        y_target_lab_i.append(Path(adversarial_dir, 'lab', name_sample).with_suffix('.lab'))

    window_size_samples = tools.next_pow2_samples(parameters['window_size'], sampling_rate)
    hop_size_samples = tools.sec_to_samples(parameters['hop_size'], sampling_rate)

    info = pd.DataFrame(index = [0], columns=['eps','predicted_word', 'real_word', 
                        'target_word', 'N_0', 'I_0' , 'D_0' , 
                        'S_0' , 'N_tar' ,'I_tar'  ,'D_tar' ,'S_tar' ,
                        'max_entropy', 'max_variance', 'max_sum_variance' , 
                        'max_aleatoric','max_sum_aleatoric', 'max_KL', 'max_MI', 'max_var2'])

    info = info.fillna(0)

    for eps in EPS_LIST: # np.arange(0.0, .101, 0.01):
        eps = np.round(eps, 2)
        I = 0
        D = 0
        S = 0
        N = 0

        I_tar = 0
        D_tar = 0
        S_tar = 0
        N_tar = 0
        with tqdm(total=len(y_target_list), file=sys.stdout, desc=f'eps = {eps:0.3}', bar_format='    {l_bar}{bar:20}{r_bar}') as pbar:

            for i, (target, y_target_label, x_test, y_test, y_test_grid) in enumerate(
                    zip(y_target_list, y_target_lab_i, x_test_i, y_test_i, y_test_grid_i)):
                
                y_tar = tools.praat_file_to_word_target(target, sampling_rate, window_size_samples, hop_size_samples, hmm)

                if multi_model:
                    mod = rec.ensemble_model(model)
                else:
                    mod = model

                name_sample = target.with_suffix('').name
                file_name = f'''{model_name}_{MODEL_PARAMETERS['num_for_adv_attack']}_{MODEL_PARAMETERS['adv_attack']}_adv_ex_{name_sample}_{MODEL_PARAMETERS['model_type']}_{num_adv}_{MODEL_PARAMETERS['num_for_adv_attack']}_{eps:0.3}.npy'''
                adv_file = Path(adversarial_dir, 'calc_adv_examples', file_name)
                adv_wav_file = Path(adversarial_dir, 'wav_adversarial', file_name).with_suffix('.wav')
                mod.eval()
                sess = tf.compat.v1.Session()
                if MODEL_PARAMETERS['CREATE_ADVERSARIAL'] or not os.path.exists(adv_wav_file):
                    a, single_advs = adv.adv_ex(mod, x_test, parameters, sampling_rate, y_tar, eps,
                                   MODEL_PARAMETERS['num_for_adv_attack'], sess, MULTI_MODEL, MODEL_PARAMETERS['adv_attack'])
                    
                    torchaudio.save(str(adv_wav_file), torch.from_numpy(a).float().cpu(), sampling_rate) 
                
                a, _ = torchaudio.load(str(adv_wav_file))
                a = a.data.numpy()

                posteriors, uncertainty_meassure, word_seq, word_start_idx, word_end_idx = test_adv_example(hmm, model, a,
                                                                                                         multi_model)

                I_0, D_0, S_0, N_0 = get_accuracy(y_test, word_seq)
                I_1, D_1, S_1, N_1 = get_accuracy(y_target_label, word_seq)

                I += I_0
                D += D_0
                S += S_0
                N += N_0

                I_tar += I_1
                D_tar += D_1
                S_tar += S_1
                N_tar += N_1
                
                y_target_seq = open(y_target_label).read().strip().split(" ")
                y_test_seq = open(y_test).read().strip().split(" ")

                info_new = {'eps': eps, 'predicted_word': f'{word_seq}', 
                            'real_word': f'{y_test_seq}', 
                            'target_word':f'{y_target_seq}', 
                            'N_0': N_0, 'I_0' : I_0, 'D_0' : D_0, 'S_0' : S_0, 
                            'N_tar': N_1, 'I_tar' : I_1, 'D_tar' : D_1, 
                            'S_tar' : S_1,'max_entropy': np.max(uncertainty_meassure['entropy']), 
                            'max_variance' : np.max(uncertainty_meassure['variance']), 
                            'max_sum_variance': np.max(np.sum(uncertainty_meassure['variance'], axis=1)),
                            'max_aleatoric': np.max(uncertainty_meassure['aleatoric']),
                            'max_sum_aleatoric': np.max(np.sum(uncertainty_meassure['aleatoric'], axis=0)),
                            'max_KL': np.max(uncertainty_meassure['KLD']), 'max_MI': np.max(uncertainty_meassure['MI']), 'max_var2':np.max(uncertainty_meassure['var2'])}

                info_new = pd.DataFrame(info_new, index = [0])
                info = info.append(info_new)

                accuracy_test = (N - I - D - S) / N
                accuracy_tar = (N_tar - I_tar - D_tar - S_tar) / N_tar
                pbar.set_description(f'eps = {eps:0.3} (Test accuracy {accuracy_test:.4} Target accuracy {accuracy_tar:.4})')
                pbar.update(1)

                sess.close()

        info.to_pickle(f'''{RESULTS_DIR}/info_{model_name}_{MODEL_PARAMETERS['num_for_adv_attack']}_{MODEL_PARAMETERS['model_type']}_{MODEL_PARAMETERS['adv_attack']}_{num_adv}_{EPS_STR}.h5''')


        print('I: {}    D: {}    S: {}    N: {}'.format(I, D, S, N))
        print(f'accuracy to true label: {(N - I - D - S) / N}')

        print('I_tar: {}    D_tar: {}    S_tar: {}    N_tar: {}'.format(I_tar, D_tar, S_tar, N_tar))
        print(f'accuracy to target label: {(N_tar - I_tar - D_tar - S_tar) / N_tar}')

        print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model_type', type=str, help='model type, e.g. NN')

    args = parser.parse_args()

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if not os.path.exists(TARGET_DIR):
        print(f'[+] PREPARE LAB FILES FOR FORCED ALIGNMENT... ')
        create_random_lab(DATA_DIR, TARGET_DIR, 2000)

        os.system(f'{ROOT_DIR}/montreal-forced-aligner/bin/mfa_align {TARGET_DIR} {ROOT_DIR}/digits.dict english {TARGET_DIR}/TextGrid')
        os.system(f'mv {TARGET_DIR}/TextGrid/lab/* {TARGET_DIR}/TextGrid/')
        os.system(f'rm -rf {TARGET_DIR}/TextGrid/lab')

    MODEL_PARAMETERS['model_type'] = args.model_type

    if MODEL_PARAMETERS['model_type'] == 'NN':
        print('[+] Modeltype: NN')
        import recognizer.torch_dnn_recognizer as rec

    elif MODEL_PARAMETERS['model_type'] == 'dropout':
        print('[+] Modeltype: dropout')
        import recognizer.dropout_recognizer as rec

    elif MODEL_PARAMETERS['model_type'] == 'BNN2':
        print('[+] Modeltype: BNN2')
        import recognizer.VMG2_recognizer as rec

    elif MODEL_PARAMETERS['model_type'] == 'ensemble':
        MULTI_MODEL = True
        print('[+] Modeltype: ensemble')
        import recognizer.ensemble_recognizer as rec

    else:
        raise ValueError('[+] Modeltype: does not exits... send request to Lea :D ')

    # hash function depends on feature extraction and num epoches
    parameter_str = str(FEATURE_PARAMETERS) + str(MODEL_PARAMETERS) + str(NUM_TRAIN)
    model_name = str(int(hashlib.sha1(parameter_str.encode('utf-8')).hexdigest(), 16) % (10 ** 16))
    hmm = HMM.HMM()

    
    print(f'[+] TRAIN OR LOAD MODEL {model_name}... ')
    model, sampling_rate, hmm = train_model(DATA_DIR, EXP_DIR, hmm, model_name, 
                                            MODEL_DIR, MODEL_PARAMETERS, 
                                            FEATURE_PARAMETERS, MULTI_MODEL,
                                            TRAIN, VITERBI_TRAIN)

    if TEST:
        print('[+] TEST MODEL... ')
        accuracy = test_model(DATA_DIR, hmm, model, MODEL_PARAMETERS, MULTI_MODEL, model_name, NUM_TEST, RESULTS_DIR, 'TEST') 
        accuracy = test_model(DATA_DIR, hmm, model, MODEL_PARAMETERS, MULTI_MODEL, model_name, NUM_TEST, RESULTS_DIR, 'TEST-FIT')

    if ADVERSARIAL:
        print('[+] CALCULATE ADVERSARIAL EXAMPLES...')
        targeted_attack_model(model, hmm, FEATURE_PARAMETERS, MULTI_MODEL, model_name, NUM_ADV, sampling_rate)
