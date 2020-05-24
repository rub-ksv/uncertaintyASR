import numpy as np
import recognizer.feature_extraction as fe
# restrict to one gpu if not distributed learning
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from cleverhans.attacks import MadryEtAl
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import CarliniWagnerL2
#todo : disable matplotlib in docker
# import matplotlib.pyplot as plt
import recognizer.tools as tools

from pathlib import Path

from time import sleep

import torchaudio

def adv_ex(model, x_in, parameters, sampling_rate, target, eps, n_adv, sess, multi_model, attack):

    if parameters['feature_type'] == 'raw':
        hop_size_samples = tools.sec_to_samples(parameters['hop_size'], sampling_rate)

        x, _ = torchaudio.load(x_in)

        num_frames = np.floor(x.shape[1]/hop_size_samples)
        x = x[:,:int(num_frames * hop_size_samples)-1]
    else:
        x = fe.compute_features_with_context(x_in, **parameters)
        x = np.reshape(x, (x.shape[0], (x.shape[1] * x.shape[2])), order='C')

    signal_length = x.shape[1]
    window_size_samples = tools.next_pow2_samples(parameters['window_size'], sampling_rate)
    hop_size_samples = tools.sec_to_samples(parameters['hop_size'], sampling_rate)

    num_frames = tools.get_num_frames(signal_length, window_size_samples, hop_size_samples) + 1

    # if target length does nt fit signal length
    if target.shape[0] != num_frames:
        x = x[:, :-hop_size_samples]
        signal_length = x.shape[1]
        num_frames = tools.get_num_frames(signal_length, window_size_samples, hop_size_samples) + 1

    adv, single_advs = targeted(model, x.shape,sess, x, target, eps, n_adv,attack,  multi_model) # x.cpu().numpy(),
    return adv, single_advs


def targeted(model, input_dim,sess, X_test,target, eps, n_adv,attack,  multi_model= False):

    '''
    Calculates adversarial examples with the projected gradient decent method by madry et al.
    :return: adversarial examples for X_test
    '''

    tf_model_fn = convert_pytorch_model_to_tf(model, out_dims=95)
    if multi_model:
        cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='probs')
    else:
        cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')
    x_op = tf.compat.v1.placeholder(tf.float32, shape=(None, input_dim[1]))
    if attack =='PGD':
        attack_op = MadryEtAl(cleverhans_model, sess=sess)
        attack_params = {'eps': eps,
                       'y_target': target,
                       'clip_min': -1,
                       'clip_max': 1}
    elif attack == 'FGSM':
        attack_op = FastGradientMethod(cleverhans_model, sess=sess)
        attack_params = {'eps': eps,
                       'y_target': target,
                       'clip_min': -1,
                       'clip_max': 1}
    elif attack == 'CWL2':
        attack_op = CarliniWagnerL2(cleverhans_model, sess=sess)
        attack_params = {'max_iterations':100,
                  'clip_min': -1,
                   'clip_max': 1}

    else:
        raise ValueError('[+] Attack not supported')

    if not os.path.exists('/root/asr-python/src/tmp2'):
        os.makedirs('/root/asr-python/src/tmp2')

    adv_x_op = attack_op.generate(x_op, **attack_params)
    m = input_dim[0] #minibatchsize
    adv_x = np.zeros([m, input_dim[1]])
    adv_samples = n_adv
    single_advs = []
    for i in range(adv_samples):
        single_adv = sess.run(adv_x_op, feed_dict={x_op: X_test})
        single_advs.append(single_adv)
        # np.save(Path('/root/asr-python/src/tmp2', f'''{i}.npy'''), single_adv)
        adv_x += (1 / adv_samples) * single_adv
    
    # np.save(Path('/root/asr-python/src/tmp2', f'''combined.npy'''), adv_x)

    # adv_x
    return adv_x, np.array(single_advs)


def plot_entropy(entropy, ref_seq, pred_seq,tar_seq, word_start_idx, word_end_idx,eps, modeltype,start_real, end_real, start_tar, end_tar,i):
    dom = np.arange(0, len(entropy), 1)
    plt.figure(figsize=(15, 5))
    plt.plot(dom, entropy , '-x', label='entropy', zorder=9999)
    plt.plot()
    top = np.max(entropy)
    me = top/2
    plt.gca().set_prop_cycle(None)
    for r in range(len(start_real)):
        plt.plot(np.array([start_real[r], end_real[r]]), np.array([0,0]), '--|', zorder=9999)
    plt.gca().set_prop_cycle(None)
    print(word_start_idx)
    print(word_end_idx)
    for r in range(len(word_start_idx)):
        plt.plot(np.array([word_start_idx[r], word_end_idx[r]]), np.array([me , me]), '--o', zorder=9999)
    plt.gca().set_prop_cycle(None)
    for r in range(len(start_tar)):
        plt.plot(np.array([start_tar[r], end_tar[r]]), np.array([top,top]), '--x', zorder=9999)
    plt.xlabel('Window number')
    plt.ylabel('Entropy')
    plt.title(f'Targ label: {tar_seq}\nPred label: {pred_seq}\nReal label: {ref_seq}', loc='left')
    plt.savefig(f'../results/{modeltype}_{eps}_{i}_entropy.png',dpi=150, bbox_inches='tight')
    plt.close()


def plot_KLD(KL, ref_seq, pred_seq,tar_seq, word_start_idx, word_end_idx,eps, modeltype, start_real, end_real, start_tar, end_tar,i):
    dom = np.arange(0, len(KL), 1)
    plt.figure(figsize=(15, 5))
    plt.plot(dom, KL ,'-x', label='entropy', zorder=9999)
    plt.plot()
    top = np.max(KL)
    me = top/2
    plt.gca().set_prop_cycle(None)
    for r in range(len(start_real)):
        plt.plot(np.array([start_real[r], end_real[r]]), np.array([0,0]), '--|', zorder=9999)
    plt.gca().set_prop_cycle(None)
    for r in range(len(word_start_idx)):
        plt.plot(np.array([word_start_idx[r], word_end_idx[r]]), np.array([me , me]), '--o', zorder=9999)
    plt.gca().set_prop_cycle(None)
    for r in range(len(start_tar)):
        plt.plot(np.array([start_tar[r], end_tar[r]]), np.array([top,top]), '--x', zorder=9999)
    plt.xlabel('Window number')
    plt.ylabel('KLD')
    plt.title(f'Targ label: {tar_seq}\nPred label: {pred_seq}\nReal label: {ref_seq}', loc='left')
    plt.savefig(f'../results/{modeltype}_{eps}_{i}_KLD.png',dpi=150, bbox_inches='tight')
    plt.close()


def plot_variance(variance, ref_seq, pred_seq, tar_seq, word_start_idx, word_end_idx, eps, modeltype,start_real, end_real, start_tar, end_tar,i):
    pic = np.transpose(variance)
    plt.figure(figsize=(15, 5))
    plt.xlabel('Window number')
    plt.ylabel('Variance')
    plt.title(f'Targ label: {tar_seq}\nPred label: {pred_seq}\nReal label: {ref_seq}', loc='left')
    plt.imshow(pic, interpolation='none')
    for r in range(len(start_real)):
        plt.plot(np.array([start_real[r], end_real[r]]), np.array([90,90]), '--|', zorder=9999)
    plt.gca().set_prop_cycle(None)
    for r in range(len(word_start_idx)):
        plt.plot(np.array([word_start_idx[r], word_end_idx[r]]), np.array([45, 45]), '--o', zorder=9999)
    plt.gca().set_prop_cycle(None)
    for r in range(len(start_tar)):
        plt.plot(np.array([start_tar[r], end_tar[r]]), np.array([0,0]), '--x', zorder=9999)
    plt.savefig(f'../results/{modeltype}_{eps}_{i}_var.png', dpi=150, bbox_inches='tight')
    plt.close()




def return_start_words(praat_file, sampling_rate = 16000, window_size =25e-3 , hop_size=12.5e-3):
    #:param praat_file: *.TextGrid file.
    window_size_samples = tools.sec_to_samples(window_size, sampling_rate)
    hop_size_samples = tools.sec_to_samples(hop_size, sampling_rate)

    intervals, min_time, max_time = tools.praat_to_word_Interval(praat_file)

    # parse intervals
    starts = []
    ends = []
    for interval in intervals:
        start_frame = tools.sec_to_frame(interval.start, sampling_rate, window_size_samples, hop_size_samples)
        end_frame = tools.sec_to_frame(interval.end, sampling_rate, window_size_samples, hop_size_samples)
        starts.append(start_frame)
        ends.append(end_frame)
    return starts, ends
