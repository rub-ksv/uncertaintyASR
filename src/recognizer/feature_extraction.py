import numpy as np
import recognizer.tools as tools
from scipy.io import wavfile
from scipy import fftpack
import random
import torch

import torchaudio


def generator(model, hmm, x_dirs, y_dirs, sampling_rate, parameters, viterbi_training=False):
    """
    creates feature-target-pairs out of files lists for training.
    :param model: trained dnn model
    :param hmm: hmm class instance
    :param x_dirs: *.wav file list
    :param y_dirs: *.TextGrid file list
    :param sampling_rate: sampling frequency in hz
    :param parameters: parameters for feature extraction
    :param viterbi_training: flag for viterbi training
    :return: x, y: feature-target-pair
    """
    # set random seed
    random.seed(42)
    # init A for viterbo training
    hmm.A_count = np.ceil(hmm.A)
    # same values for all utterances
    window_size_samples = tools.next_pow2_samples(parameters['window_size'], sampling_rate)
    hop_size_samples = tools.sec_to_samples(parameters['hop_size'], sampling_rate)
    # generator
    while True:
        x_dirs, y_dirs = tools.shuffle_list(x_dirs, y_dirs)
        for audio_file, target_dir in zip(x_dirs, y_dirs):
            # get features and target
            y = tools.praat_file_to_word_target(target_dir, sampling_rate, window_size_samples, hop_size_samples, hmm)
            x, _ = torchaudio.load(audio_file)
            
            # to have the same number of frames as the targets
            num_frames = np.floor(x.shape[1]/hop_size_samples)
            x = x[:,:int(num_frames * hop_size_samples)-1]
 
            yield x, y, target_dir


def roll(x, n):  
    return torch.cat((x[-n:], x[:-n]))


def add_context_pytorch(feats, left_context=4, right_context=4):
    """
    Adds context to the features.

    :param feats: extracted features.
    :param left_context: Number of predecessors.
    :param right_context: Number of succcessors.
    :return: Features with context.
    """

    feats_context = feats.unsqueeze(2)
    for i in range(1, left_context + 1): 
        tmp = roll(feats, i).unsqueeze(2)
        feats_context = torch.cat((tmp, feats_context), 2)
    
    for i in range(1, right_context + 1):
        tmp = roll(feats, -i).unsqueeze(2)
        feats_context = torch.cat((feats_context, tmp), 2)

    return feats_context
