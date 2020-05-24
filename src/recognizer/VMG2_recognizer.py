import recognizer.feature_extraction as fe
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
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

    NUM_PREDICTIONS = 5
    if advanced_eval:
        y_pred = []
        for r in range(NUM_PREDICTIONS):
            y_i = model.forward(x, False)
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
            y_i = model.forward(x, False)
            y_i = F.softmax(y_i, dim=1).cpu().data.numpy()
            y += 1 / NUM_PREDICTIONS * y_i

    return y, uncertainty_meassure


def train_model(model, x_dirs, y_dirs, hmm, sampling_rate, parameters, steps_per_epoch=10, epochs=10,
                viterbi_training=False):
    """
    Train DNN with Bayesian Perspectiv

    :param model: trained dnn model
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

    N = 1505426
    S = 5 # Samples for Expectation
    lr = 0.0001 # learning Rate

    opt = optim.Adam(model.parameters(), lr)

    for n_iter in range(epochs):
        i = 0
        with tqdm(total=steps_per_epoch, file=sys.stdout, desc='Loss', bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:

            ac_loss = 0
            length = 0
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
                    t_mb = torch.max(t_mb, 1)[1]
                    E_theta_nll = 0
                    for _ in range(S):
                        y_s, D_KL = model.forward(X_mb, True)
                        # sometimes the targets dimensions differ (by one frame)
                        if t_mb.shape[0] != y_s.shape[0]:
                            diff = y_s.shape[0] - t_mb.shape[0]
                            y_s = y_s[:t_mb.shape[0],:]
                            if diff > 1:
                                raise ValueError('Frame difference larger than 1!')

                        y_cat = dists.Categorical(logits=y_s).log_prob(t_mb)
                        E_theta_nll += 1 / S * y_cat

                    E_nll = -torch.mean(E_theta_nll)
                    loss = E_nll + (1/ N) * D_KL

                    ac_loss += loss.item()
                    length += len(local_X)

                    av_loss = ac_loss/length

                    #if i % 10 == 0:
                    pbar.set_description(f'Epochs {n_iter+1}/{epochs} (loss {av_loss:.6})')

                    pbar.update(1)
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
                    torch.cuda.empty_cache()

    return model


def dnn_model(output_shape, left_context, right_context, n_mfcc, h_dim=512):
    """
    Creates DNN and returns it.

    :param input_shape: shape of input data
    :param output_shape: shape of output data

    :return untrained DNN
    """

    class VMGLinear(nn.Module):
        ''' Layer for a Bayesian Neural Net'''

        def __init__(self, in_dim, out_dim):
            super(VMGLinear, self).__init__()

            in_dim += 1
            self.in_dim = in_dim
            self.out_dim = out_dim

            self.mu = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(in_dim, out_dim)))
            self.logvar_in = nn.Parameter(nn.init.normal_(torch.zeros(in_dim), -9, 1e-6))
            self.logvar_out = nn.Parameter(nn.init.normal_(torch.zeros(out_dim), -9, 1e-6))

        def forward(self, x):
            m = x.size(0)
            W, D_KL = self.sample(m)
            x = torch.cat([x, torch.ones(m, 1, device='cuda')], 1)
            h = torch.bmm(x.unsqueeze(1), W).squeeze(1)

            return h, D_KL

        def sample(self, m):
            r, c = self.in_dim, self.out_dim
            M = self.mu
            # ToDo : softvar = log (1+exp(x))
            logvar_r = self.logvar_in
            logvar_c = self.logvar_out
            var_r = torch.exp(logvar_r)
            var_c = torch.exp(logvar_c)

            E = torch.randn(m, *M.shape, device='cuda')

            # Reparametrization trick
            W = M + torch.sqrt(var_r).view(1, r, 1) * E * torch.sqrt(var_c).view(1, 1, c)

            # KL divergence to prior MVN(0, I, I)
            D_KL = 1 / 2 * (torch.sum(var_r) * torch.sum(var_c) \
                            + torch.norm(M) ** 2 \
                            - r * c - c * torch.sum(logvar_r) - r * torch.sum(logvar_c))

            return W, D_KL

    class VMGNet(nn.Module):

        def __init__(self, output_dim, left_context=4, right_context=4, n_mfcc=13, h1_dim=100, h2_dim=100):
            super(VMGNet, self).__init__()

            self.left_context = left_context
            self.right_context = right_context
            self.n_mfcc = n_mfcc

            self.mfcc = torchaudio.transforms.MFCC(n_mfcc = self.n_mfcc)
            
            self.deltas = torchaudio.transforms.ComputeDeltas()
            self.fc_xh1 = VMGLinear(3 * self.n_mfcc * (self.left_context + self.right_context + 1), h1_dim)
            self.fc_h1h2 = VMGLinear(h1_dim, h2_dim)
            self.fc_h2y = VMGLinear(h2_dim, output_dim)

        def forward(self, x, returnKL=False):
            x = x/torch.max(torch.abs(x))

            # calc mfcc
            mfcc = self.mfcc(x)
            
            # add delta and delta deltas
            deltas = self.deltas(mfcc)
            deltadeltas = self.deltas(deltas)
            mfcc = torch.cat((mfcc, deltas, deltadeltas), 1)

            mfcc = torch.squeeze(mfcc).permute(1,0)

            mfcc_context = fe.add_context_pytorch(mfcc, self.left_context, self.right_context)

            h1, D_KL1 = self.fc_xh1(torch.flatten(mfcc_context, start_dim=1))
            h1 = F.relu(h1)
            h2, D_KL2 = self.fc_h1h2(h1)
            h2 = F.relu(h2)
            y, D_KL3 = self.fc_h2y(h2)
            return (y, D_KL1 + D_KL2 + D_KL3) if returnKL else y

    model = VMGNet(output_shape).cuda()

    return model
