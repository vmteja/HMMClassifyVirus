# -*- coding: utf-8 -*-
#@ Divya , Krishna & Tarun
# python code to find out emission and transition probabilty matrices
from __future__ import division
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from random import randint

a=[];


# S = Start State, E = Exit State, V = Viral state, NV = Non Viral State
states = ["V", "NV"]
n_states = len(states)

emissions = ["a", "t", "c", "g"]
n_emissions = len(emissions)

# Convert character emission symbols to integers as 
# required by MultinomialHMM

discrete_emissions = LabelEncoder().fit_transform(emissions)
print "LabelEncoder().fit_transform(", emissions, ") = ", discrete_emissions

# Create discrete emission dictionary
emission_dict = dict(zip(emissions, discrete_emissions))
print "emission_dict = ", emission_dict


start_probability = np.array([  1.00000000e+000,   3.74827284e-109])

transition_probability = np.array([
 [ 0.58270126,  0.41729874],
 [ 0.36767417,  0.63232583]

])

emission_probability = np.array([
 [ 0.18405977,  0.38473778,  0.01781628,  0.41338617],
 [ 0.50532274,  0.02042444,  0.40648306,  0.06776975]
])

#model = hmm.MultinomialHMM(n_components = n_states, params = "ste", init_params = "ste", n_iter = 50)
#model = hmm.MultinomialHMM(n_components = n_states, params = "ste", n_iter = 10000 , init_params='')
model = hmm.MultinomialHMM(n_components = n_states, params = "ste", n_iter = 10000 , init_params='')

model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability
print "Parameters before fitting the model:"
print
print model.startprob_
print model.transmat_
print model.emissionprob_
print 

file1='banana_all'
print file1
# Read in sample sequence from file
sample_seq1 = []
with open('C:\\teja\\bio\\temp\\'+file1, 'r') as f:
    # Read in entire sequence in file and remove all carriage returns "\n"
    sample_seq1 = list(f.read().replace('\n', ''))
    f.closed

# Convert sample seq to integer numbers per emission_dict
sample_seq2 = [emission_dict[e] for e in sample_seq1]

# print sample_seq2
#print "sample_seq after conversion = ", sample_seq

sample_seq = np.array([sample_seq2]).T

"""
Estimate model parameters.
An initialization step is performed before entering the EM algorithm. 
If you want to avoid this step, set the keyword argument init_params 
to the empty string ‘’. Likewise, if you would like just to do an initialization, 
call this method with n_iter=0.
"""
model = model.fit(sample_seq)

print ">>> Get parameters for the estimator."
model_parameters = model.get_params(deep = True)
print "Model Parameters = ", model_parameters
print 
print "Parameters after fitting the model:"
print model.startprob_
print model.transmat_
print model.emissionprob_

print

"""logprob, hidden_model = model.decode(sample_seq, algorithm="viterbi")
# print "Sample Sequence (Characters:", sample_seq1
# print "Length Of Sample Sequence = ", len(sample_seq1)
# print "Sample Sequence (Integers): ", sample_seq2
# print "Hidden Model (i.e., from  Viterbi Algorithm):", ", ".join(map(lambda x: states[x], hidden_model))
# print

print ">>> Compute the log probability under the model."
model_score = model.score(sample_seq)
print "Model Score = ", model_score
print

print ">>> Compute the model posteriors (using Forward & Backward algorithm)."
print ">>> Just last posterior probs are printed by default. "
print ">>> To print all posterior probs, change _PRINT_ALL_ in source code to True."
_PRINT_ALL_ = False
model_posterior_probs = model.predict_proba(sample_seq)
if _PRINT_ALL_:
    print "Model Posterior Probss = ", model_posterior_probs
else:
    print "Last Model Posterior Prob = ", model_posterior_probs[len(model_posterior_probs)-1]
print "Length of Posteriors = ", len(model_posterior_probs)
print

print ">>> Find most likely state sequence corresponding to observed emissions using the Baum-Welch algorithm."
model_seq = model.predict(sample_seq)
print "Model Most Likely State Seq = ", model_seq
print
"""