import numpy as np

def calculate_variance(tensor):
    '''gets a numpy ndarray of shape NUM_Samples, NumberWindows, 95 '''
    variance = []
    tensor = np.array(tensor)
    for r in range(tensor.shape[1]):
        var = np.var(tensor[:, r, :], axis=0)
        variance.append(var)
    return np.array(variance)

def calculate_entropy(tensor):
    '''gets a numpy ndarray of shape NUM_Samples, NumberWindows, 95 '''
    entropy = []
    tensor = np.array(tensor)
    final = tensor.mean(0)
    for r in range(final.shape[0]):
        ent_cal = (-final[r, :] * np.log(final[r, :] + 1e-8)).sum()
        entropy.append(ent_cal)
    return np.array(entropy)

def calculate_KLD(tensor):
    '''gets a numpy ndarray of shape NUM_Samples, NumberWindows, 95 '''
    KL = np.zeros((tensor.shape[1]))
    for j in range(tensor.shape[1]):
        for r in range(tensor.shape[0]-1):
            KL[j] += 1/(tensor.shape[0]-1)*compute_kl_divergence(tensor[r,j,:], tensor[r+1,j,:])
    return KL


def compute_kl_divergence(p_probs, q_probs):
    """"KL (p || q)"""
    p_probs = np.clip(p_probs, 1e-6, 1)
    q_probs = np.clip(q_probs, 1e-6, 1)
    kl_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(kl_div)

# def calculate_aleatoric_uncertainty(tensor):
#     '''Calculates the uncertainty as (4) in https://openreview.net/pdf?id=Sk_P2Q9sG
#     gets a numpy ndarray of shape: number of drawn samples, number of different pictures put in, output nodes (10)
#     '''
#     aleat = []
#     for r in range(tensor.shape[1]):
#         aleatoric = np.mean(tensor[:, r, :] * (1 - tensor[:, r, :]), axis=0)
#         aleat.append(aleatoric)
#     return aleat

def calculate_aleatoric_uncertainty(tensor):
    '''Calculates the uncertainty as (4) in https://openreview.net/pdf?id=Sk_P2Q9sG
    gets a numpy ndarray of shape: number of drawn samples, number of different pictures put in, output nodes (10)
    '''
    aleat = []
    tensor = np.array(tensor)
    final = tensor.mean(0)
    for r in range(final.shape[0]):
        aleatoric = final[r, :] * (1 - final[r, :])
        aleat.append(aleatoric)
    return aleat

def calculate_feinmann_variance(tensor):
    '''gets a numpy ndarray of shape NUM_Samples, NumberWindows, 95 '''
    variance = []
    tensor = np.array(tensor)
    final = tensor.mean(0) 
    for r in range(tensor.shape[1]):
        yty = 0
        for s in range(tensor.shape[0]):
            # yty += 1/ (tensor.shape[0]) * np.mutmul(tensor[s,r,:], tensor[s,r,:])
            yty += 1/ (tensor.shape[0]) * (tensor[s,r,:] * tensor[s,r,:])
        var = yty - final[r,:]**2
        variance.append(var)
    return np.array(variance)

def calculate_mutual_information(tensor):
    '''gets a numpy ndarray of shape NUM_Samples, NumberWindows, 95 '''
    mi = []
    tensor = np.array(tensor)
    final = tensor.mean(0)
    for r in range(tensor.shape[1]):
        ent_cal = (-final[r, :] * np.log(final[r, :] + 1e-8)).sum()
        ErwEnt = 0
        for s in range(tensor.shape[0]):
            ErwEnt += 1/ (tensor.shape[0]) * ((-tensor[s,r, :] * np.log(tensor[s,r, :] + 1e-8)).sum())
        MI = ent_cal - ErwEnt
        mi.append(MI)
    return np.array(mi)