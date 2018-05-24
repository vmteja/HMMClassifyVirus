# -*- coding: utf-8 -*-
#@ Divya , Krishna & Tarun
# python code to find out log prob scores of genomes of banana streak family virus
from __future__ import division
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from random import randint

a=[];

modelScores={};

def sequencer1(number):
	# S = Start State, E = Exit State, V = Viral state, NV = Non Viral State
	states = ["V", "NV"]
	n_states = len(states)

	emissions = ["a", "t", "c", "g"]
	n_emissions = len(emissions)

	# Convert character emission symbols to integers as 
	# required by MultinomialHMM

	discrete_emissions = LabelEncoder().fit_transform(emissions)
	#print "LabelEncoder().fit_transform(", emissions, ") = ", discrete_emissions

	# Create discrete emission dictionary
	emission_dict = dict(zip(emissions, discrete_emissions))
	#print "emission_dict = ", emission_dict


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
	model = hmm.MultinomialHMM(n_components = n_states, params = "", n_iter = 0 , init_params='')

	model.startprob_ = start_probability
	model.transmat_ = transition_probability
	model.emissionprob_ = emission_probability
	#print "Parameters before fitting the model:"
	#print
	#print model.startprob_
	#print model.transmat_
	#print model.emissionprob_
	#print 

	if number==1:
	    file1='bac1'
	elif number==2:
	    file1='banana_ca'
	elif number==3:
	    file1='banana_ua'
	elif number==4:
	    file1='banana_gf'
	elif number==5:
	    file1='banana_im'
	elif number==6:
	    file1='banana_my'
	elif number==7:
	    file1='banana_ol'
	elif number==8:
	    file1='banana_ua'
	elif number==9:
	    file1='banana_ui'
	elif number==10:
	    file1='banana_um'
	elif number==11:
	    file1='banana_ul'
	#print file1
	# Read in sample sequence from file
	sample_seq1 = []
	with open('C:\\teja\\bio\\temp\\'+file1, 'r') as f: 
	    # Read in entire sequence in file and remove all carriage returns "\n"
	    sample_seq1 = list(f.read().replace('\n', ''))
	    f.closed

	#print "sample_seq before conversion = ", sample_seq

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

	#print ">>> Get parameters for the estimator."    last
	model_parameters = model.get_params(deep = True)
	#print "Model Parameters = ", model_parameters    last
	#print 
	#print "Parameters after fitting the model:"
	#print model.startprob_
	#print model.transmat_
	#print model.emissionprob_

	#print

	logprob, hidden_model = model.decode(sample_seq, algorithm="viterbi")
	# print "Sample Sequence (Characters:", sample_seq1
	# print "Length Of Sample Sequence = ", len(sample_seq1)
	# print "Sample Sequence (Integers): ", sample_seq2
	# print "Hidden Model (i.e., from  Viterbi Algorithm):", ", ".join(map(lambda x: states[x], hidden_model))
	# print

	#print ">>> Compute the log probability under the model."
	model_score = model.score(sample_seq)
	#print "Model Score for  ",file1,"   is =", repr(model_score)
	#print
	a.append(model_score);
	modelScores[file1]=model_score;
'''
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
'''
sequencer1(3);
sequencer1(4);
sequencer1(5);
sequencer1(6);
sequencer1(7);
sequencer1(8);
sequencer1(9);
sequencer1(10);
sequencer1(11);
sequencer1(2);
#sequencer1(1)
print 

print "Calculating log prob scores for the different viruses present in the banana streak family"
print
for i in modelScores:
	print i,"  has score of ",modelScores[i]

#print modelScores	