import numpy as np
import recognizer.feature_extraction as fe
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchaudio

def viterbi_train(x, y, target_dir, model, hmm):

    x_tmp = x.data.numpy()

    tmp = features_to_posteriors(model, x_tmp)

    stateSequence, _ = hmm.viterbi_decode(tmp)

    word_seq, _, _  = hmm.getTranscription(stateSequence)

    lab_dir = str(target_dir).replace('TextGrid', 'lab')
    ref_seq = open(lab_dir).read().strip().split(' ')

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
    y = features_to_posteriors(model, x)

    return x, y if returnx else y


def features_to_posteriors(model, features):
    """
    Calculates posteriors for audio file.

    :param model: trained dnn model
    :param audio_file: *.wav file
    :param parameters: parameters for feature extraction

    :return: posteriors
    """
    x = torch.from_numpy(features).float().cuda()
    y = model.forward(x).cpu()
    y = F.softmax(y, dim=1).data.numpy()
    return y


def train_model(models, x_dirs, y_dirs, hmm, sampling_rate, parameters, steps_per_epoch=10, epochs=10,
                viterbi_training=False):

    lr = 0.0001 #learning rate
    lam = 1/1505426 # weight decay
    
    opts = [optim.Adam(model.parameters(), lr, weight_decay=lam) for model in models]
    CrEnt = nn.CrossEntropyLoss()
    for n_iter in range(epochs):
        for model_n, (model, opt) in enumerate(zip(models, opts)):
            i = 0
            
            ac_loss = 0
            length = 0
            with tqdm(total=steps_per_epoch, file=sys.stdout, desc='Loss', bar_format='{l_bar}{bar:30}{r_bar}') as pbar:

                for local_X, local_y, target_dir in fe.generator(model, hmm, x_dirs, y_dirs, sampling_rate, parameters):

                    if viterbi_training:
                        local_X, local_y = viterbi_train(local_X, local_y, target_dir, model, hmm)

                    if (i > steps_per_epoch):
                        break
                    else:
                        i += 1

                        X_mb = local_X.cuda()
                        X_mb = X_mb.view(X_mb.size()[0], -1).float()
                        t_mb = torch.from_numpy(local_y).long().cuda()
                        y = model.forward(X_mb)
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

                        pbar.set_description(f'Model {model_n + 1} Epochs {n_iter+1}/{epochs} (loss {av_loss:.6})')
                            
                        pbar.update(1)
                        loss.backward()
                        nn.utils.clip_grad_value_(model.parameters(), 5)
                        opt.step()
                        opt.zero_grad()
                        torch.cuda.empty_cache()
    return models


def dnn_model(output_shape, left_context, right_context, n_mfcc):
    """
    Creates DNN and returns it.

    :param output_shape: shape of output data

    :return untrained DNN
    """

    # define model
    class Model(nn.Module):

        def __init__(self, output_shape, left_context=4, right_context=4, n_mfcc=13):
            super(Model, self).__init__()

            self.left_context = left_context
            self.right_context = right_context
            self.n_mfcc = n_mfcc

            self.mfcc = torchaudio.transforms.MFCC(n_mfcc = self.n_mfcc)
            
            self.deltas = torchaudio.transforms.ComputeDeltas()
            
            self.fcxh1 = nn.Linear(3 * self.n_mfcc * (self.left_context + self.right_context + 1), 100)
            self.fch1h2 = nn.Linear(100, 100)
            self.fch2y = nn.Linear(100, output_shape)

        def forward(self, x):
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
            h2 = F.relu(self.fch1h2(h1))
            y = self.fch2y(h2)
            return y

    model = Model(output_shape).cuda()

    return model


class ensemble_model(nn.Module):
    def __init__(self, models):
        super(ensemble_model, self).__init__()
        self.model0 = models[0]
        self.model1 = models[1]
        self.model2 = models[2]
        self.model3 = models[3]
        self.model4 = models[4]

    def forward(self,x):
        y0 = self.model0(x.clone())
        y0 = F.softmax(y0, dim=1)
        y1 = self.model1(x.clone())
        y1 = F.softmax(y1, dim=1)
        y2 = self.model2(x.clone())
        y2 = F.softmax(y2, dim=1)
        y3 = self.model3(x.clone())
        y3 = F.softmax(y3, dim=1)
        y4 = self.model4(x.clone())
        y4 = F.softmax(y4, dim=1)
        y = 1/5*(y0+y1+y2+y3+y4)
        return y