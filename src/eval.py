import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm

np.random.seed(2020)

ROOT_DIR = 'root/asr-python/'
RESULTS_DIR = Path(ROOT_DIR, 'results')


MODELS = {'NN': '1178423951416274', 'dropout': '9633085383667387', 'BNN2': '7928232086334539', 'ensemble': '9023427500653420'}

EVAL_PARAMETERS = {'attack': 'PGD',
                   'attack_samples':1}

def load_data(model_name, set_name, value):
    '''
    loads test datasets
    if something is correctly classified it gets a 1
    if something was attacked it gets a 1
    '''
    df = pd.read_pickle(Path(RESULTS_DIR, f'''info_{set_name}_{value}_{model_name}_1000.h5'''))
    df = df.iloc[1:,:].reset_index()
    df['correct'] = 0
    df.loc[df['predicted_word'] == df['real_word'], 'correct'] = 1
    df['eps']= 0
    df['attacked'] = 0
    return df

def load_attacked_data(model, attack, NUM_ADV, eps, value):
    '''if something is correctly classified it gets a 1
    if something was attacked it gets a 1
    '''
    df = pd.read_pickle(Path(RESULTS_DIR, f'''info_{value}_{EVAL_PARAMETERS['attack_samples']}_{model}_{attack}_{NUM_ADV}_{eps}.h5'''))
    df = df.iloc[1:,:].reset_index()
    df['correct'] = 0
    df.loc[df['predicted_word'] == df['real_word'], 'correct'] = 1
    df['attacked'] = 1
    df.loc[df['eps'] == 0, 'attacked'] = 0
    return df

def calc_ROC(y_pred, y_real):
    fpr, tpr, threshold = roc_curve(y_real, y_pred)
    roc_auc= auc(fpr, tpr)
    print(f'AUROC Score: {"%0.3f" % roc_auc}')


def calculate_accuracy(subframe, adv=True):
    S_0 = subframe['S_0']
    I_0 = subframe['I_0']
    D_0 = subframe['D_0']
    N_0 = subframe['N_0']
    S_0 = sum(S_0)
    I_0 = sum(I_0)
    D_0 = sum(D_0)
    N_0 = sum(N_0)
    acc_real = (N_0 - I_0 - D_0 - S_0) / N_0

    if adv == True:
        S_tar = subframe['S_tar']
        I_tar = subframe['I_tar']
        D_tar = subframe['D_tar']
        N_tar = subframe['N_tar']
        S_tar = sum(S_tar)
        I_tar = sum(I_tar)
        D_tar = sum(D_tar)
        N_tar = sum(N_tar)
        acc_tar = (N_tar - I_tar - D_tar - S_tar) / N_tar

    return (acc_real, acc_tar) if adv else acc_real



if __name__ == "__main__":

    #for mod in MODELS:
    for mod, value in MODELS.items():
        print('\n###################################')
        print(f'\n[+] Evaluation for model {mod}')
        print('###################################\n')
        print('[++] Accuracies')
        df_from_test = load_data(mod, 'test_data_TEST', value)
        print(f'accuracy test set: {calculate_accuracy(df_from_test, False)}')
        df_fit = load_data(mod, 'test_data_TEST-FIT', value)
        print(f'accuracy test set 2 : {calculate_accuracy(df_fit, False)}')
        df = load_attacked_data(mod, EVAL_PARAMETERS['attack'], 100, '0.0-0.1', value)
        acc_tar= []
        acc_real= []
        '''print('[+++] 100 samples with varying epsilon')
        for e in df['eps'].unique():

            a_r, a_t = calculate_accuracy(df.loc[df['eps'] == e, :])
            print(f'eps:  {"%0.2f" % e} __ acc to tar : {"%0.3f" % a_t }  __ acc to real : {"%0.3f" % a_r}')'''

        print('[+++] 1000 samples with fixed epsilon')
        df_05 = load_attacked_data(mod, EVAL_PARAMETERS['attack'], 1000, '0.05', value)
        for e in df_05['eps'].unique():
            a_r, a_t = calculate_accuracy(df_05.loc[df_05['eps'] == e, :])
            print(f'eps:  {e} __ acc to tar : {"%0.3f" % a_t }  __ acc to real : {"%0.3f" % a_r}')

        '''df_02 = load_attacked_data(mod, EVAL_PARAMETERS['attack'], 1000, '0.02', value)
        for e in df_02['eps'].unique():
            a_r, a_t = calculate_accuracy(df_02.loc[df_02['eps'] == e, :])
            print(f'eps:  {e} __ acc to tar : {"%0.3f" % a_t }  __ acc to real : {"%0.3f" % a_r}')'''

        print('\n[++] Feature evaluation')

        feature = ['max_KL', 'max_entropy', 'max_variance', 'max_MI']
        for df in [df_05]: #, df_02]:
            print(f'''[+++] attacked with {df['eps'].unique()}''')
            for fea in feature:
                print(f'Feature: {fea}')

                X_test = df_from_test
                X_test = X_test.append(df, sort=False)
                X_all = X_test[fea]

                # Learn the parameters of the Gaussian distribution on test data 2
                parameters = norm.fit(df_fit[fea])

                # Get the probability density of the test set and attacked set under the previously learned gaussian distribution
                fitted_norm = norm.pdf(X_all, loc=parameters[0], scale=parameters[1]+1e-08)
                calc_ROC(fitted_norm, np.abs(X_test['attacked'] - 1))


