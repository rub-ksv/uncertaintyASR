import math
import numpy as np
import recognizer.tools as tools
import recognizer.feature_extraction as fe
import os
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import recognizer.hmm as HMM
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import torch.optim as optim
import torch.utils.data as dat 

import sys
from tqdm import tqdm

import torchaudio

def viterbi_train(x, y, target_dir, model, hmm):

    x_tmp = x.data.numpy()

    tmp, _ = features_to_posteriors(model, x_tmp)

    stateSequence, _ = hmm.viterbi_decode(tmp) 

    word_seq, _, _  = hmm.getTranscription(stateSequence)

    lab_dir = str(target_dir).replace('TextGrid', 'lab')
    ref_seq = open(lab_dir).read().strip().split(' ')

    # apply viterbi training only if transcription is correct
    if np.array_equal(ref_seq, word_seq):
        hmm.A_count, y = hmm.viterbiTraining(stateSequence, hmm.A_count)

    return x, y



def wav_to_posteriors(model, audio_file, returnx=False):
    """
    Calculates posteriors for audio file.

    :param model: trained dnn model
    :param audio_file: *.wav file
    :param parameters: parameters for feature extraction

    :return: posteriors
    """
    x, _ = torchaudio.load(audio_file)
    
    x = x.data.numpy()
    y, uncertainty_meassure = features_to_posteriors(model, x)

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

    y = model.forward(x)
    y = F.softmax(y, dim=1).cpu().data.numpy()
    entropy = []
    aleatoric = []
    for r in range(y.shape[0]):
        ent_cal = (-y[r,:]*np.log(y[r,:] + 1e-8)).sum(0)
        entropy.append(ent_cal)
        alea = y[r, :] * (1 - y[r, :])
        aleatoric.append(alea)
    variance = np.ones((y.shape[0], 95))

    KL= np.zeros(y.shape[0])
    MI = np.zeros(y.shape[0])
    var2 = np.zeros(y.shape[0])

    uncertainty_meassure = {'entropy': entropy,
                            'variance': variance,
                            'aleatoric': aleatoric,
                            'var2': var2,
                            'MI': MI,
                            'KLD': KL}

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

    opt = optim.Adam(model.parameters(), lr, weight_decay=lam)
    CrEnt = nn.CrossEntropyLoss()

    for n_iter in range(epochs):
        i=0
        # progress bar
        with tqdm(total=steps_per_epoch, file=sys.stdout, desc='Loss', bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:

            ac_loss = 0
            length = 0
            for local_X, local_y, target_dir in fe.generator(model, hmm, x_dirs, y_dirs, sampling_rate, parameters, viterbi_training):
                
                if viterbi_training:
                    local_X, local_y = viterbi_train(local_X, local_y, target_dir, model, hmm)

                if(i>steps_per_epoch):
                    break
                else:

                    i +=1
                    X_mb = local_X.cuda() 
                    par_prev = model.fch2y.weight.cpu().detach().numpy()

                    y = model.forward(X_mb)
                    t_mb = torch.from_numpy(local_y).long().cuda()
                    ref = torch.max(t_mb , 1)[1]

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

                    # progress bar description
                    pbar.set_description(f'Epochs {n_iter+1}/{epochs} (loss {av_loss:.6})')

                    # update progress bar
                    pbar.update(1)

                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
                    torch.cuda.empty_cache()
    return model


def dnn_model(output_shape, left_context, right_context, n_mfcc):
    """
    Creates DNN and returns it.

    :param output_shape: shape of output data

    :return untrained DNN
    """
    class Model(nn.Module):

        def __init__(self, output_shape, left_context=4, right_context=4, n_mfcc=13):
            super(Model, self).__init__()

            self.left_context = left_context
            self.right_context = right_context
            self.n_mfcc = n_mfcc

            # mfcc
            self.mfcc = torchaudio.transforms.MFCC(n_mfcc = self.n_mfcc)
            
            # delta and deltadeltas
            self.deltas = torchaudio.transforms.ComputeDeltas()

            self.fcxh1 = nn.Linear(3 * self.n_mfcc * (self.left_context + self.right_context + 1), 100)
            self.fch1h2 = nn.Linear(100, 100)
            self.fch2y = nn.Linear(100, output_shape)

        def forward(self, x):
            # normalize input
            x = x/torch.max(torch.abs(x))

            # calc mfcc
            mfcc = self.mfcc(x)
            
            # add delta and delta deltas
            deltas = self.deltas(mfcc)
            deltadeltas = self.deltas(deltas)
            mfcc = torch.cat((mfcc, deltas, deltadeltas), 1)

            mfcc = torch.squeeze(mfcc).permute(1,0)

            mfcc_context = fe.add_context_pytorch(mfcc, self.left_context, self.right_context)
            
            h1 = self.fcxh1(torch.flatten(mfcc_context, start_dim=1))
            h1 = F.relu(h1)
            h2 = self.fch1h2(h1)
            h2 = F.relu(h2)
            y = self.fch2y(h2)
            return y
    
    model = Model(output_shape).cuda()

    return model
