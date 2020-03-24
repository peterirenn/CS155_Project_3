import numpy as np
########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.
        L = self.L      # Local copies of class parameters
        A = self.A
        O = self.O

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M)]
        seqs = [['' for _ in range(self.L)] for _ in range(M)]

        # Initialize base cases
        for y in range(L):
            probs[0][y] = self.A_start[y] * self.O[y][x[0]]
            seqs[0][y] = str(y)

        # Run viterbi
        for k in range(M-1): 
            for y in range(L):
                (prob,state) = max((probs[k][y0] * A[y0][y] * O[y][x[k+1]],y0) for y0 in range(L))
                probs[k+1][y] = prob
                seqs[k+1][y] = seqs[k][state] + str(y)

        # Find maximum probability sequence
        k += 1
        (prob,state) = max((probs[k][y],y) for y in range(L))
        max_seq = seqs[k][state]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[1. for _ in range(self.L)] for _ in range(M + 1)]
        L = self.L      # Local copies of class parameters
        A = self.A
        O = self.O

        # Initialize base cases
        for y in range(L): alphas[1][y] = O[y][x[0]]*self.A_start[y]
        if normalize: alphas[1] = [alphas[1][y] / sum(alphas[1]) for y in range(L)]

        # Run forward algorithm
        for k in range(1,M): 
            for y in range(L):
                alphas[k+1][y] = sum(O[y][x[k]]*alphas[k][y0]*A[y0][y] for y0 in range(L))
            if normalize: 
                alphas[k+1] = [alphas[k+1][y] / sum(alphas[k+1]) for y in range(L)]
        
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        L = self.L      # Local copies of class parameters
        A = self.A
        O = self.O

        # Initialize base cases
        for y in range(L): betas[M][y] = 1.
        if normalize: betas[M] = [betas[M][y] / sum(betas[M]) for y in range(L)]

        # Run backward algorithm
        for k in range(M,0,-1):
            for y in range(L):
                betas[k-1][y] = sum(betas[k][y1]*A[y][y1]*O[y1][x[k-1]] for y1 in range(L))
            if normalize:
                betas[k-1] = [betas[k-1][y] / sum(betas[k-1]) for y in range(L)]
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''
        
        # Calculate each element of A using the M-step formulas.
        for a in range(self.L):
            for b in range(self.L):
                num_sum, den_sum = 0.,0.
                for j in range(len(Y)):
                    for i in range(0,len(Y[j])-1):
                        if Y[j][i] == a: 
                            den_sum += 1
                        if Y[j][i] == a and Y[j][i+1] == b: 
                            num_sum += 1
                self.A[a][b] = num_sum / den_sum

        # Calculate each element of O using the M-step formulas.
        for w in range(self.D):
            for z in range(self.L): 
                num_sum, den_sum = 0.,0.
                for j in range(len(Y)): 
                    for i in range(len(Y[j])): 
                        if Y[j][i] == z: 
                            den_sum += 1
                            if X[j][i] == w: 
                                num_sum += 1
                self.O[z][w] = num_sum / den_sum


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        for iteration in range(N_iters):
            #if iteration % 10 == 0: print('current iteration: '+str(iteration))
            print('Current iteration: ' + str(iteration))
            A_num, A_den = [[0. for _ in range(self.L)] for _ in range(self.L)], [[0. for _ in range(self.L)] for _ in range(self.L)]
            O_num, O_den = [[0. for _ in range(self.D)] for _ in range(self.L)], [[0. for _ in range(self.D)] for _ in range(self.L)]
            for j in range(len(X)): 
                M = len(X[j])
                # Initialize probs
                probs = [[0. for _ in range(self.L)] for _ in range(M)]
                probs_joint = [[[0. for _ in range(self.L)] for _ in range(self.L)] for _ in range(M)]

                ### Expectation step
                # Compute alphas
                alphas = self.forward(X[j],normalize=True)
                # Compute betas
                betas = self.backward(X[j],normalize=True)

                # Compute probabilities
                for i in range(0,M):
                    # Compute marginals
                    probs[i] = [alphas[i+1][z]*betas[i+1][z] / sum([alphas[i+1][s]*betas[i+1][s] for s in range(self.L)]) for z in range(self.L)]
                for i in range(0,M-1):
                    # Compute joint marginals
                    den_sum = sum([sum([alphas[i+1][a] * betas[i+2][b] * self.A[a][b] * self.O[b][X[j][i+1]] for a in range(self.L)]) for b in range(self.L)])
                    for a in range(self.L): 
                        for b in range(self.L):
                            probs_joint[i][a][b] = alphas[i+1][a] * betas[i+2][b] * self.A[a][b] * self.O[b][X[j][i+1]] / den_sum
                    #probs_joint[i] = [[alphas[i][a] * betas[i+1][b] * self.A[a][b] * self.O[b][X[j][i]] / den_sum for a in range(self.L)] for b in range(self.L)]

                ### Maximization step
                # Update transition matrix using marginals
                for a in range(self.L):
                    for b in range(self.L):
                        A_num[a][b] += sum([probs_joint[i][a][b] for i in range(M-1)])
                        A_den[a][b] += sum([probs[i][a] for i in range(M-1)])
                # Update observation matrix using marginals
                for w in range(self.D):
                    for z in range(self.L): 
                        O_den[z][w] += sum([probs[i][z] for i in range(M)])
                        for i in range(M): 
                            if X[j][i] == w: 
                                O_num[z][w] += probs[i][z]

            # Update transition and observation matrices
            self.A = [[A_num[a][b] / A_den[a][b] for b in range(self.L)] for a in range(self.L)]
            self.O = [[O_num[a][b] / O_den[a][b] for b in range(self.D)] for a in range(self.L)]


    def generate_emission(self, M, syl_map):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate in syllables.
            syl_map:    Map from emission to number of syllables

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        import numpy as np

        emission = []
        states = []

        states.append(int(np.random.choice(self.L,1)))
        emission.append(int(np.random.choice(self.D,1,p=self.O[int(states[0])])))
        n_syls = syl_map[emission[0]]
        # Generate states
        while n_syls < M: 
            too_long = True
            while too_long: 
                next_state = int(np.random.choice(self.L, 1, p = self.A[int(states[-1])]))
                next_emission = int(np.random.choice(self.D,1,p=self.O[int(states[-1])]))
                if n_syls + syl_map[next_emission] <= M: too_long = False
            states.append(next_state)
            emission.append(next_emission)
            n_syls += syl_map[emission[-1]]

        return emission, states, n_syls

    def generate_sonnet(self, M, syl_map, rhyme_pairs):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate in syllables.
            syl_map:    Map from emission to number of syllables

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        import numpy as np

        emission = []
        states = []

        for line_num in range(14): 
            cur_emission = []
            cur_states = []
            cur_states.append(int(np.random.choice(self.L,1)))
            cur_emission.append(int(np.random.choice(self.D,1,p=self.O[int(cur_states[0])])))
            n_syls = syl_map[cur_emission[0]]
            # Generate states
            while n_syls < M: 
                next_state = int(np.random.choice(self.L, 1, p = self.A[int(cur_states[-1])])[0])
                next_emission = int(np.random.choice(self.D,1,p=self.O[next_state])[0])
                n_tries = 0
                if n_syls + syl_map[next_emission] >= M: 
                    if n_tries > self.D: return emission,1,1
                    n_tries += 1
                    # Cases for rhyming lines
                    if line_num in [2,3,6,7,10,11,13]: 
                        if line_num == 13: rhyme_num = line_num-1
                        else: rhyme_num = line_num-2
                        final_word_options = self.get_rhymes(emission[rhyme_num][-1],rhyme_pairs,M-n_syls,syl_map)
                        # Case where no rhyme is possible
                        while len(final_word_options) == 0: 
                            rhymes_a = self.get_rhymable_words_of_len(syl_map[emission[rhyme_num][-1]],syl_map,rhyme_pairs)
                            emission[rhyme_num][-1] = self.emission_from_options(states[rhyme_num][-1],rhymes_a)
                            final_word_options = self.get_rhymes(emission[rhyme_num][-1],rhyme_pairs,M-n_syls,syl_map)

                    # Cases for non rhyming lines
                    else:  
                        final_word_options = self.get_rhymable_words_of_len(M-n_syls,syl_map,rhyme_pairs)

                    # Generate emission
                    next_emission = self.emission_from_options(next_state,final_word_options)
                    n_syls += syl_map[next_emission]

                # Append emission, state to current row
                cur_states.append(next_state)
                cur_emission.append(next_emission)
                n_syls += syl_map[cur_emission[-1]]

            # Append row to poem
            emission.append(cur_emission)
            states.append(cur_states)

        return emission, states, n_syls

    def emission_from_options(self,state,final_word_options): 
        probs = [self.O[state][word] for word in final_word_options]
        probs_sum = sum(probs)
        probs = [p/probs_sum for p in probs]
        return int(np.random.choice(final_word_options,1,p=probs)[0])

    def get_rhymable_words_of_len(self,n_syls,syl_map,rhyme_pairs): 
        words_of_len = self.get_words_of_len(n_syls,syl_map)
        rhymers = []
        for word in words_of_len: 
            if self.rhymable(word,rhyme_pairs): rhymers.append(word)
        return rhymers

    def get_rhymes(self,cur_word,rhyme_pairs,n_syls,syl_map): 
        rhymes = []
        for pair in rhyme_pairs: 
            if pair[0] == cur_word and syl_map[pair[1]] == n_syls: 
                rhymes.append(pair[1])
        return rhymes

    def get_words_of_len(self,n_syls,syl_map):
        words = []
        for word in range(self.D): 
            if syl_map[word] == n_syls: words.append(word)
        return words

    def rhymable(self,word,rhyme_pairs): 
        for pair in rhyme_pairs: 
            if pair[0] == word: return True
        return False

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
