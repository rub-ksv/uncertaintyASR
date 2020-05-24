import numpy as np
from praatio import tgio
import random


def sec_to_samples(x, sampling_rate):
    """
    Converts continuous time to sample index.

    :param x: scalar value representing a point in time in seconds.
    :param samskypling_rate: sampling rate in Hz.
    :return: sample index.
    """
    return int(x * sampling_rate)


def next_pow2(x):
    """
    Returns the next power of two for any given positive number.

    :param x: scalar input number.
    :return: next power of two larger than input number.
    """
    return int(np.ceil(np.log2(np.abs(x))))


def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):
    """
    Returns the total number of frames for a given signal length with corresponding window and hop sizes.

    :param signal_length_samples: total number of samples.
    :param window_size_samples: window size in samples.
    :param hop_size_samples: hop size (frame shift) in samples.
    :return: total number of frames.
    """
    overlap_samples = window_size_samples - hop_size_samples

    return int(np.ceil(float(signal_length_samples - overlap_samples) / hop_size_samples))


def hz_to_mel(x):
    """
    Converts a frequency given in Hz into the corresponding Mel frequency.

    :param x: input frequency in Hz.
    :return: frequency in mel-scale.
    """
    return 2595 * np.log10(1 + float(x) / 700)


def mel_to_hz(x):
    """
    Converts a frequency given in Mel back into the linear frequency domain in Hz.

    :param x: input frequency in mel.
    :return: frequency in Hz.
    """
    return 700 * (10**(x / 2595) - 1)


def next_pow2_samples(x, sampling_rate):
    """
    Returns the next power of two in number of samples for a given length in seconds and sampling rate

    :param x: length in seconds.
    :sampling_rate: sampling rate in Hz.
    :return: next larger power of two in number of samples
    """
    # return 2**next_pow2(sec_to_samples(x, sampling_rate))
    return sec_to_samples(x, sampling_rate)


def sample_to_frame(x, window_size_samples, hop_size_samples):
    """
    converts sample index to frame index.

    :param x: sample index.
    :param window_size_samples: window length in samples.
    :param hop_size_samples:    hop length in samples
    :return: frame index.
    """
    return int(np.floor(x / hop_size_samples))


def sec_to_frame(x, sampling_rate, window_size_samples, hop_size_samples):
    """
    Converts time in seconds to frame index.

    :param x:  time in seconds
    :param sampling_rate:  sampling frequency in hz
    :param window_size_samples: window length in samples.
    :param hop_size_samples:    hop length in samples
    :return: frame index
    """
    return sample_to_frame(sec_to_samples(x, sampling_rate), window_size_samples, hop_size_samples)


def divide_interval(num, start, end):
    """
    Divides the number of states equally to the number of frames in the interval.

    :param num:  number of states.
    :param start: start frame index
    :param end: end frame index
    :return starts: start indexes
    :return end: end indexes
    """
    interval_size = end - start
    # gets remainder 
    remainder = interval_size % num
    # init sate count per state with min value
    count = [int((interval_size - remainder)/num)] * num
    # the remainder is assigned to the first n states
    count[:remainder] = [x + 1 for x in count[:remainder]]
    # init starts with first start value
    starts = [start]
    ends = [] 
    # iterate over the states and sets start and end values
    for c in count[:-1]:
        ends.append(starts[-1] + c)
        starts.append(ends[-1])

    # set last end value
    ends.append(starts[-1] + count[-1])

    return starts, ends


def praat_file_to_word_target(praat_file, sampling_rate, window_size_samples, hop_size_samples, hmm):
    """
    Reads in praat file and calculates the phone-based target matrix.

    :param praat_file: *.TextGrid file.
    :param sampling_rate: sampling frequency in hz
    :param window_size_samples: window length in samples
    :param hop_size_samples: hop length in samples
    :return: target matrix for DNN training
    """
    # gets list of intervals, start, end, and word/phone
    intervals, min_time, max_time = praat_to_word_Interval(praat_file)

    # we assume min_time always to be 0, if not, we have to take care of this
    if not min_time == 0:
        raise Exception("Houston we have a problem: start value of audio file is not 0 for file: {}".format(praat_file))

    # gets dimensions of target
    max_sample = sec_to_samples(max_time, sampling_rate)
    num_frames = get_num_frames(max_sample, window_size_samples, hop_size_samples)
    num_states = hmm.get_num_states()

    # init target with zeros
    target = np.zeros((num_frames, num_states))

    # parse intervals
    for interval in intervals:
        # get state index, start and end frame
        states = hmm.input_to_state(interval.label)
        start_frame = sec_to_frame(interval.start, sampling_rate, window_size_samples, hop_size_samples)
        end_frame = sec_to_frame(interval.end, sampling_rate, window_size_samples, hop_size_samples)

        # divide the interval equally to all states
        starts, ends = divide_interval(len(states), start_frame, end_frame)

        # assign one-hot-encoding to all segments of the interval
        for state, start, end in zip(states, starts, ends):    
            # set state from start to end to 1
            target[start:end, state] = 1

    # find all columns with only zeros...
    zero_column_idxs = np.argwhere(np.amax(target, axis=1) == 0)
    # ...and set all as silent state
    target[zero_column_idxs, hmm.input_to_state('sil')] = 1

    return target

        
def praat_file_to_phone_target(praat_file, sampling_rate, window_size_samples, hop_size_samples, hmm):
    """
    Reads in praat file and calculates the phone-based target matrix.

    :param praat_file: *.TextGrid file.
    :param sampling_rate: sampling frequency in hz
    :param window_size_samples: window length in samples
    :param hop_size_samples: hop length in samples
    :return: target matrix for DNN training
    """
    # gets list of intervals, start, end, and word/phone
    intervals, min_time, max_time = praat_to_phone_Interval(praat_file)

    # we assume min_time always to be 0, if not, we have to take care of this
    if not min_time == 0:
        raise Exception("Houston we have a problem: start value of audio file is not 0 for file: {}".format(praat_file))

    # gets dimensions of target
    max_sample = sec_to_samples(max_time, sampling_rate)
    num_frames = get_num_frames(max_sample, window_size_samples, hop_size_samples)
    num_states = hmm.get_num_states()
    # init target with zeros
    target = np.zeros((num_frames, num_states))

    # parse intervals
    for interval in intervals:
        # get state index, start and end frame
        states = hmm.input_to_state(interval.label)
        start_frame = sec_to_frame(interval.start, sampling_rate, window_size_samples, hop_size_samples)
        end_frame = sec_to_frame(interval.end, sampling_rate, window_size_samples, hop_size_samples)

        # divide the interval equally to all states
        starts, ends = divide_interval(len(states), start_frame, end_frame)

        # assign one-hot-encoding to all segments of the interval
        for state, start, end in zip(states, starts, ends):    
            # set state from start to end to 1
            target[start:end, state] = 1

    # find all columns with only zeros...
    zero_column_idxs = np.argwhere(np.amax(target, axis=1) == 0)
    # ...and set all as silent state
    target[zero_column_idxs, hmm.input_to_state('sil')] = 1

    return target


def praat_to_word_Interval(praat_file):
    """
    Reads in one praat file and returns interval description.

    :param praat_file: *.TextGrid file path

    :return itervals: returns list of intervals, 
                        containing start and end time and the corresponding word/phobe.
    :return min_time:    min timestamp of audio (should be 0)
    :return max_time:    min timestamp of audio (should be audio length)
    """
    # read in praat file (expects one *.TextGrid file path)
    tg = tgio.openTextgrid(praat_file)

    # read return values
    itervals = tg.tierDict['words'].entryList
    min_time = tg.minTimestamp
    max_time = tg.maxTimestamp

    # we will read in word-based
    return itervals, min_time, max_time


def praat_to_phone_Interval(praat_file):
    """
    Reads in one praat file and returns interval description.

    :param praat_file: *.TextGrid file path

    :return itervals: returns list of intervals, 
                        containing start and end time and the corresponding word/phone.
    :return min_time: min timestamp of audio (should be 0)
    :return max_time: min timestamp of audio (should be audio length)
    """
    # read in praat file (expects one *.TextGrid file path)
    tg = tgio.openTextgrid(praat_file)

    # read return values
    itervals = tg.tierDict['phones'].entryList
    min_time = tg.minTimestamp
    max_time = tg.maxTimestamp

    # we will read in word-based
    return itervals, min_time, max_time


def shuffle_list(*ls):
    """
    Shuffles all list in ls with same permutation

    :param ls: list of list to shuffle.
    :return: shuffled lists.
    """
    l =list(zip(*ls))

    random.shuffle(l)
    return zip(*l)
