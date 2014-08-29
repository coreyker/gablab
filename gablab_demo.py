# GabLab demo
# Analysis with a multi-scale Gabor dictionary

import numpy as np
import pylab
from gablab import *
from scikits import audiolab, samplerate

fs = 8000. # sample rate

# Load an audio file
y,fstmp,fmt = audiolab.wavread('glockenspiel.wav')
#y = np.sum(y,1)/y.shape[1] # make mono
y = samplerate.resample(y, fs/fstmp,'sinc_best') # resample for faster computation

# setup
scales = [256,512,1024,2048, 4096] # window sizes to use
L = np.ceil(len(y)/float(np.max(scales)))*np.max(scales) # calc block boundary
y = np.hstack((y,np.zeros(L-len(y)))) # pad input to block boundary

# Build multi-scale Gabor dictionary
D = DictionaryUnion(*[GaborBlock(L,s) for s in scales])

# Analysis (basis pursuit denoising: min ||x||_1 s.t. ||y-Dx||_2<e)
x = BPDN(D,y,maxerr=1e-12,maxits=1000)

# Plots
parts = D.parts(x)

for i,p in enumerate(parts):
    pylab.figure()
    freqsup = range(0,D[i].fftLen/2+1)
    pylab.imshow(20*np.log10(np.abs(p[:,freqsup].transpose())), aspect='auto', interpolation='bilinear', origin='lower')

# Sounds
synth = [np.array(samplerate.resample(np.real(D[i].dot(p)), fstmp/fs, 'sinc_best'), dtype='float64') for i,p in enumerate(parts)]

for s in synth:
    audiolab.play(s)

pylab.show()
