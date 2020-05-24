import recognizer.feature_extraction as fe
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import recognizer.functions as fx
import sys
from tqdm import tqdm
import torchaudio


def viterbi_train(x, y, target_dir, model, hmm):

    x_tmp = x.data.numpy()

    tmp, _ = features_to_posteriors(model, x_tmp)

    stateSequence, _ = hmm.viterbi_decode(tmp)

    word_seq, _, _ = hmm.getTranscription(stateSequence)

    lab_dir = str(target_dir).replace('TextGrid', 'lab')
    ref_seq = open(lab_dir).read().strip().split(' ')

    if np.array_equal(ref_seq, word_seq):
        hmm.A_count, y = hmm.viterbiTraining(stateSequence, hmm.A_count)

    return x, y


def wav_to_posteriors(model, audio_file, returnx = True):
    """
    Calculates posteriors for audio file.

    :param model: trained dnn model
    :param audio_file: *.wav file
    :param parameters: parameters for feature extraction
    :return: posteriors
    """
    x, _ = torchaudio.load(audio_file)

    x = x.data.numpy()

    y, uncertainty_meassure = features_to_posteriors(model, x, returnx)

    return x, y, uncertainty_meassure if returnx else y


def features_to_posteriors(model, features, advanced_eval=False):
    """
    Calculates posteriors for audio file.

    :param model: trained dnn model
    :param audio_file: *.wav file
    :param parameters: parameters for feature extraction

    :return: posteriors
    """
    x = torch.from_numpy(features).float().cuda()
    y = 0

    uncertainty_meassure = {'entropy': 0,
                            'variance': 0,
                            'aleatoric': 0,
                            'var2': 0,
                            'MI': 0,
                            'KLD': 0}

    NUM_PREDICTIONS = 100
    if advanced_eval:
        y_pred = []
        for r in range(NUM_PREDICTIONS):
            y_i = model.forward(x)
            y_i = F.softmax(y_i, dim=1).cpu().data.numpy()
            y_pred.append(y_i)
            y += 1 / NUM_PREDICTIONS * y_i
        y_pred = np.array(y_pred)
        uncertainty_meassure['entropy'] = fx.calculate_entropy(y_pred)
        uncertainty_meassure['variance'] = fx.calculate_variance(y_pred)
        uncertainty_meassure['aleatoric'] = fx.calculate_aleatoric_uncertainty(y_pred)
        uncertainty_meassure['var2'] = fx.calculate_feinmann_variance(y_pred)
        uncertainty_meassure['MI'] = fx.calculate_mutual_information(y_pred)
        uncertainty_meassure['KLD'] = fx.calculate_KLD(y_pred)
    else:
        for r in range(NUM_PREDICTIONS):
            y_i = model.forward(x)
            y_i = F.softmax(y_i, dim=1).cpu().data.numpy()
            y += 1 / NUM_PREDICTIONS * y_i

    return y, uncertainty_meassure 


def train_model(model, x_dirs, y_dirs, hmm, sampling_rate, parameters, steps_per_epoch=10, epochs=10, viterbi_training=False):
    """
    Train DNN.

    :param model: trained dnn model
    :param model_dir: directory for model
    :param x_dirs: *.wav file list
    :param y_dirs: *.TextGrid file list
    :param hmm: hmm class instance
    :param sampling_rate: sampling frequency in hz
    :param parameters: parameters dictionary for feature extraction 
    :param steps_per_epoch: steps per epoch
    :param param steps_per_epoch: steps per epoch
    :param viterbi_training: flag for viterbi training

    :return model: trained model
    :return history: training history
    """

    lr = 0.0001 #learning rate
    lam = 1/1505426 # weight decay

    opt = optim.Adam(model.parameters(), lr, weight_decay=lam) #, weight_decay=lam)
    #opt = optim.Adadelta(model.parameters(), lr=1)

    CrEnt = nn.CrossEntropyLoss()

    # if viterbi_training:
    #     w2 = model.fch1h2.weight.cpu() /2
    #     w3 = model.fch2y.weight.cpu() /2
    #
    #     model.fch1h2.weight.data =w2
    #     model.fch2y.weight.data =w3
    #     model.cuda()


    for n_iter in range(epochs):
        i = 0

        ac_loss = 0
        length = 0
        # progress bar
        with tqdm(total=steps_per_epoch, file=sys.stdout, desc='Loss', bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:

            for local_X, local_y, target_dir in fe.generator(model, hmm, x_dirs, y_dirs, sampling_rate, parameters):

                if viterbi_training:
                    local_X, local_y = viterbi_train(local_X, local_y, target_dir, model, hmm)

                if(i>steps_per_epoch):
                    break
                else:
                    i +=1
                    
                    X_mb = local_X.cuda() 

                    # X_mb = torch.from_numpy(local_X).cuda()
                    X_mb = X_mb.view( X_mb.size()[0],-1).float()
                    t_mb = torch.from_numpy(local_y).long().cuda()
                    y = 0
                    for r in range(10):
                        y += 1/10 * model.forward(X_mb)
                    ref = torch.max(t_mb, 1)[1]

                    # sometimes the targets dimensions differ (by one frame)
                    if ref.shape[0] != y.shape[0]:
                        diff = y.shape[0] - ref.shape[0]
                        y = y[:ref.shape[0],:]
                        
                        if diff > 1:
                            raise ValueError('Frame difference larger than 1!')

                    loss = CrEnt(y, ref)

                    ac_loss += loss.item()
                    length += len(local_X)

                    av_loss = ac_loss/length

                    #if i % 10 == 0:
                    pbar.set_description(f'Epochs {n_iter+1}/{epochs} (loss {av_loss:.6})')

                    # update progress bar
                    pbar.update(1)

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 1)
                    opt.step()
                    torch.cuda.empty_cache()


    return model


def dnn_model(output_shape, left_context, right_context, n_mfcc):
    """
    Creates DNN and returns it.

    :param output_shape: shape of output data

    :return untrained DNN
    """
    # define dropout model     
    
    class Model(nn.Module):

        def __init__(self, output_shape, left_context=4, right_context=4, n_mfcc=13):
            super(Model, self).__init__()

            self.n_mfcc = n_mfcc
            self.left_context = left_context
            self.right_context = right_context
            self.n_mfcc = n_mfcc

            self.mfcc = torchaudio.transforms.MFCC(n_mfcc = self.n_mfcc)
            
            self.deltas = torchaudio.transforms.ComputeDeltas()

            self.fcxh1 = nn.Linear(3 * self.n_mfcc * (self.left_context + self.right_context + 1), 100)
            self.fch1h2 = nn.Linear(100, 100)
            self.fch2y = nn.Linear(100, output_shape)

        def forward(self, x):

            dropout = True
            x = x/torch.max(torch.abs(x))
            
            # calc mfcc
            mfcc = self.mfcc(x)
            
            # add delta and delta deltas
            deltas = self.deltas(mfcc)
            deltadeltas = self.deltas(deltas)
            mfcc = torch.cat((mfcc, deltas, deltadeltas), 1)

            mfcc = torch.squeeze(mfcc).permute(1,0)
            
            mfcc_context = fe.add_context_pytorch(mfcc, self.left_context, self.right_context)

            h1 = F.relu(self.fcxh1(torch.flatten(mfcc_context, start_dim=1)))
            h1 = F.dropout(h1, p=0.5, training=dropout)
            h2 = F.relu(self.fch1h2(h1))
            h2 = F.dropout(h2, p=0.5, training=dropout)
            y = self.fch2y(h2)
            return y

    model = Model(output_shape).cuda()

    return model