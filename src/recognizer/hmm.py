import numpy as np

# default option
WORDS = {
    "name": ["sil", "oh", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    # "size":[    1,   3,    15,   12,    6,     9,      9,     9,   12,     15,      6,    9 ],
    "size": [1, 6, 10, 10, 6, 9, 9, 9, 10, 10, 6, 9],
    # "gram":[   100,  10,   100,  100,  100,   100,    100,   100,   10,    100,    100,  100 ],
    "gram": [100000, 1, 100000, 100000, 100000, 100000, 100000, 100000, 10000, 100000, 100000, 100000],
}


class HMM:
    iBm = []
    ini = []
    ent = []
    esc = []
    A_count = []

    def __init__(self, words=WORDS):

        self.words = words
        self.words["gram"] = [x / sum(self.words["gram"]) for x in self.words["gram"]]

        N = sum(self.words["size"])
        self.iBm = [x for x in range(N)]

        a = np.cumsum(self.words["size"])
        self.ini = list(zip([0] + a[:-1].tolist(), self.words["gram"]))
        self.ent = list(zip(a[0:], self.words["name"][1:]))
        self.esc = list(zip(a[1:] - 1, self.words["name"][1:]))

        # init Pi vector with zeros, first sil state = 0
        self.piVec = [0] * len(self.iBm)
        self.piVec[self.words['name'].index('sil')] = 1

        self.A = np.zeros((N, N))
        # basic L/R transition structure
        for n in range(0, N):
            self.A[n, n] = 0.5
            self.A[n - 1, n] = 0.5

        self.A = self.modifyTransitions(self.A)

    def get_num_states(self):
        return sum(self.words["size"])

    def modifyTransitions(self, A):

        rowSum = A.sum(axis=1, keepdims=True)
        np.seterr(divide='ignore', invalid='ignore')
        A[:] = np.where(rowSum > 0, A / rowSum, 0.);

        a = np.cumsum(self.words["size"]).tolist()
        a00 = A[0][0]  # keep sil
        for row in [x - 1 for x in a]:  # esc
            aii = A[row][row]  # self-transition
            for col, scr in zip([0] + a[:-1], self.words["gram"]):  # ini
                A[row][col] = (1.0 - aii) * scr  # remaining prob

        A[0][0] = a00 * (1. + self.words["gram"][0])  # repair sil

        return A

    def input_to_state(self, input):
        """ maps phone to corresponding HMM state

        Input:

        input:  phone


        Output:

        itervals:   HMM state of input

        """
        if input not in self.words['name']:
            raise Exception("Undefined word/phone: {}".format(input))

        # start index of each word
        start_idx = np.insert(np.cumsum(self.words["size"]), 0, 0)

        idx = self.words['name'].index(input) + 1

        start_state = start_idx[idx - 1]
        end_state = start_idx[idx]

        return [n for n in range(start_state, end_state)]

    def limLog(self, x):
        MINLOG = 1e-1000
        return np.log(np.maximum(x, MINLOG))

    def viterbi_decode(self, posteriors):  # function [best_path,Pstar] = viterbi_ue7(features,HMM)
        logPi = self.limLog(self.piVec)
        logA = self.limLog(self.A)

        # print(logA)

        logpost = self.limLog(posteriors)
        logLike = logpost[:, self.iBm]

        N = len(logPi)  # numstates = length(HMM.PI);
        T = len(logLike)  # [dim,T] = size(features);
        phi = -float("inf") * np.ones((T, N))  # phi = zeros(numstates,T);
        psi = np.zeros((T, N)).astype(int)  # psi = zeros(numstates,T);
        best_path = np.zeros(T).astype(int)

        # Initialisierung
        phi[0, :] = logPi + logLike[0]

        '''print(phi[0, :]) 
        print(psi[0, :]) 

        print('--' * 20)
        print(phi[1, :]) 
        print(psi[1, :]) 
        print('--' * 20)'''
        # Iteration
        for t in range(1, T):  # for t = 2:T
            for j in range(N):  # for j = 1:numstates
                mx = -float("inf")  # mv = -Inf;
                for i in range(N):  # for i = 1:numstates
                    if phi[t - 1, i] + logA[i, j] > mx:  # if phi(i,t-1)+HMM.A(i,j)>mv
                        mx = phi[t - 1, i] + logA[i, j];  # mx = phi(i,t-1)+HMM.A(i,j);
                        phi[t, j] = mx  # phi(j,t)=mv;
                        psi[t, j] = i  # psi(j,t)=i;
                        #       end
                        #     end
            ''' if t == 1:
                print(phi[1, :]) 
                print('--' * 20)'''
            phi[t, :] += logLike[t]  # phi(j,t)=phi(j,t)+limlog_ue7(likelihood_ue6(features(:,t),HMM.states(j).theta));

            '''if t == 1:
                print(phi[1, :]) 
                print('--' * 20)'''

        # Finde Optimum in letzter Spalte
        t = T - 1
        iopt = 0
        pstar = -float('inf')  # mv = -Inf;
        for n in range(N):  # for i = 1:numstates
            if phi[t, n] > pstar:  # if phi(i,T)>mv
                pstar = phi[t, n]  # mv = phi(i,T);
                iopt = n;  # iopt = i;
                #   end
                # end
        # Backtracking
        best_path[t] = iopt;  # statesequence(T)=iopt;
        while t > 1:  # for t = T:-1:2
            #        print( t, iopt )
            iopt = psi[t, iopt];  # iopt = psi(iopt,t);
            best_path[t - 1] = iopt;  # statesequence(t-1)=iopt;
            t = t - 1  # end
            # best_path = statesequence;
        # Ergebnis
        return best_path, pstar

    # get the word sequence for a state sequence and supress state(0) and non-exit states'''
    # exit states is a dictionary with exitState index as key and model name as value
    def getTranscription(self, statesequence):
        '''get the word sequence for a state sequence and supress state(0) and non-exit states'''
        word_list = [''] * len(self.iBm)

        ent = []
        esc = []
        for e_ent, e_esc in zip(self.ent, self.esc):
            word_list[e_esc[0]] = e_esc[1]
            # entry states
            esc.append(e_esc[0])
            # exit states
            ent.append(e_ent[0])

        words = []
        word_start_idx = []
        prev = -1
        for i, state in enumerate(statesequence):
            if state != prev:  # remove duplicate states
                if word_list[state] != '':  # if state is a valid exit state
                    words.append(word_list[state].upper())
                
                # find all entry state indexes
                if state in ent:
                    word_start_idx.append(i)

                prev = state  # remember last exit state

        word_end_idx = []
        prev = -1
        for i, state in enumerate(reversed(statesequence)):
            if state != prev:  # remove duplicate states
                # find all exit state indexes
                if state in esc:
                    word_end_idx.append(len(statesequence) - (i + 1))

                prev = state  # remember last exit state
        
        # revers end indexes and fix length
        word_end_idx = word_end_idx[::-1][:len(words)]
        word_start_idx = word_start_idx[:len(words)]

        # check if first word start index < first word end index
        if len(words) > 0:
            if word_start_idx[0] > word_end_idx[0]:
                raise AssertionError('Wrong word bounderies')
        
        return words, word_start_idx, word_end_idx


    def viterbiTraining(self, stateSequence, A):
        '''get the viterbi-posteriors for the statesequence,
        accumulate transition counts in A (extended transition matrix [S+1][S+1])'''

        posteriors = np.zeros((len(stateSequence), len(A)))
        t = 0  # frame index
        prev = 0  # initial state
        for state in stateSequence:
            posteriors[t, state] = 1  # mark posterior on best path
            A[prev, state] += 1  # increase transition count
            prev = state
            t += 1

        A[state, 0] += 1  # exit prob
        return A, posteriors
