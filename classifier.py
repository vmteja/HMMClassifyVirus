# -*- coding: utf-8 -*-
#@ Divya , Krishna & Tarun
#python code to classify the given genome sample
from __future__ import division
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from random import randint

a=[-10160.759645659991, -9854.6688234384092, -10548.848933065055, -10354.96150559862, -9976.5512483052662, -10160.759645659991, -10021.278671786104, -10152.615172103988, -9941.4248731083626, -9974.8446092331978]


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


start_probability = np.array([  1.00000000e+000,   2.20549994e-110])

transition_probability = np.array([
 [ 0.58285317,  0.41714683],
     [ 0.36782358,  0.63217642]

])

emission_probability = np.array([
 [ 0.18423615,  0.38471624,  0.0178134,   0.4132342 ],
     [ 0.50528317,  0.02031193,  0.4066259,   0.067779  ]
])

#model = hmm.MultinomialHMM(n_components = n_states, params = "ste", init_params = "ste", n_iter = 50)
#model = hmm.MultinomialHMM(n_components = n_states, params = "ste", n_iter = 10000 , init_params='')
model = hmm.MultinomialHMM(n_components = n_states, params = "", n_iter = 10000 , init_params='')

model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability
print "Parameters of the model:"
print
print model.startprob_
print model.transmat_
print model.emissionprob_
print 

file1='bac1' # change the file name to the file name you want to change to
print "Input file used : ",file1
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

logprob, hidden_model = model.decode(sample_seq, algorithm="viterbi")

print ">>> Compute the log probability under the model."
model_score = model.score(sample_seq)
print "Model Score = ", model_score
print 

if model_score in a:
	print "The Given sample belongs to banana streak family"
else:
	print "The Given sample doesn't belong to banana streak family"

