import os,sys,glob,pdb
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from gablab import *

rc('text', usetex=True)
rc('font', family='serif')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
rc('font', **font)

# dictionary
fs = 8000.
L = 4096
D1 = GaborBlock(L, 256, 4, 2)
D2 = GaborBlock(L, 32, 4, 2)
D = DictionaryUnion(D1,D2)

# create random test signal
toneSigMap = np.zeros(D1.N, dtype='complex').reshape((D1.fftLen, D1.N/D1.fftLen))
tranSigMap = np.zeros(D2.N, dtype='complex').reshape((D2.fftLen, D2.N/D2.fftLen))

np.random.seed(10)
s = 5e-2
N  = D1.N / 50 # number of non-zero coeffs
prob = 0.98 # state-transition prob

##############################
# Create synthetic test signal
##############################

# generate tonal significance map
tSteps  = D1.N/D1.fftLen
maxLen  = int(0.8*tSteps)
fftBins = np.random.permutation(D1.fftLen/2+1)
n = 0

while n<N:
    for k in fftBins:
        conjBin = (D1.fftLen - k) % D1.fftLen

        # choose random starting point for chain
        t = np.random.randint(tSteps)

        l = 0
        while np.random.binomial(1, prob) and n<N and l<maxLen:
            amp = (1 + s*np.random.randn(1))[0]
            #phase = np.exp(1j * np.pi * np.random.rand(1))[0]
            phase = np.exp(1j * 2 * np.pi * k / D1.fftLen * t * D1.stride / fs)

            if toneSigMap[k, t] == 0:
                toneSigMap[k, t] = amp*phase
                n += 1

                if conjBin != k: 
                    toneSigMap[conjBin, t] = amp*np.conj(phase)
                    n += 1

            t = (t + 1) % tSteps
            l += 1            

        if n==N: 
            break

# generate transient significance map
tSteps  = D2.N/D2.fftLen
maxLen  = int(0.8*D2.fftLen/2)
timeLocs = np.random.permutation(tSteps)
N = D2.N / 50#100
n = 0

while n<N:
    for t in timeLocs:
        
        # choose random starting bin for chain
        k = np.random.randint(D2.fftLen/2+1)

        l = 0
        while np.random.binomial(1, prob) and n<N and l<maxLen:
            amp = (1 + s*np.random.randn(1))[0]
            phase = np.exp(1j * np.pi * np.random.rand(1))[0]
            #phase = np.exp(1j * 2 * np.pi * k / D2.fftLen * t * D2.stride / fs)

            if tranSigMap[k, t] == 0:
                tranSigMap[k, t] = amp*phase
                n += 1
                
                conjBin = (D2.fftLen - k) % D2.fftLen
                if conjBin != k: 
                    tranSigMap[conjBin, t] = amp*np.conj(phase)
                    n += 1
                            
            k = (k + 1) % (D2.fftLen/2)
            l += 1        

        if n==N: 
            break

tonemap  = np.reshape(range(D1.N),(D1.N/D1.fftLen,D1.fftLen)).transpose()
transmap = np.reshape(range(D2.N),(D2.N/D2.fftLen,D2.fftLen)).transpose()

#########################
# Compute decomposition
#########################

thresh_range = np.arange(-6,0,0.1)
gamma = [0.25, 0.5, 0.75, 1]
xe = len(gamma) * [None]
TP = len(gamma) * [None]
FP = len(gamma) * [None]

nvar       = 1e-12
maxits     = 1000
stoptol    = 5e-3
muinit     = 1e-1
momentum   = 0.9
smoothinit = 1e-4
anneal     = 0.96

trans = 1

for i,g in enumerate(gamma):
    if not trans:
        f,fgrad = Tone_factory(tonemap, g)
        x = toneSigMap.transpose().flatten()
        z = np.real(D1.dot(x))
        xe[i] = GBPDN_momentum(D1,z,f,fgrad,maxerr=nvar,maxits=maxits,stoptol=stoptol,muinit=muinit,momentum=momentum,smoothinit=smoothinit,anneal=anneal)
    else:        
        f,fgrad = TT_factory(tonemap,transmap,g,g)
        x = np.hstack( ( toneSigMap.transpose().flatten(), tranSigMap.transpose().flatten() ) )
        z = np.real(D.dot(x))
        xe[i] = GBPDN_momentum(D,z,f,fgrad,maxerr=nvar,maxits=maxits,stoptol=stoptol,muinit=muinit,momentum=momentum,smoothinit=smoothinit,anneal=anneal)

    pos_sup = np.abs(x)>0
    neg_sup = np.abs(x)==0

for i in xrange(len(gamma)):    
    # measure TP and FP rate of support recovery after thresholding out small coeffs    
    TP[i] = []
    FP[i] = []
    for p in thresh_range:
    	thresh = 10**p
    	TP[i].append( np.sum((np.abs(xe[i])>thresh)*pos_sup) / float(np.sum(pos_sup)) )
    	FP[i].append( np.sum((np.abs(xe[i])>thresh)*neg_sup) / float(np.sum(neg_sup)) )

#########################
# Make ROC plot
#########################
plt.ion()
fig2 = plt.figure(figsize=(8,8))
#ax = plt.subplot()

lw = [3,3,6,6]
ls = ['-',':','-.','-']
alpha = [1,.8,.6,.4]
for i in xrange(len(gamma)):
    plt.plot(FP[i],TP[i], lw=lw[i], ls=ls[i], color='black', alpha=alpha[i])
    
leg_str = [r'$\gamma=%2.2f$' % g for g in gamma]
plt.legend(leg_str,loc=4)
plt.title('ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid('on')

##########################
# Make support plots (best points on ROC curve)
###########################
# el=1 for gamma=0.5
fig = plt.figure(figsize=(8,8))

ax = plt.subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax.set_xlabel('Time index')
ax.set_ylabel('Frequency index')

interp = 'bilinear'

ax2 = fig.add_subplot(321)
ax2.set_title('True tonal map')
ax2.set_xticklabels(())
im = plt.imshow(np.abs(toneSigMap[:D1.fftLen/2+1,:])>0, aspect='auto', interpolation=interp)
im.set_cmap('binary')

if trans:
    ax2 = fig.add_subplot(322)
    ax2.set_title('True transient map')
    ax2.set_xticklabels(())
    im = plt.imshow(np.abs(tranSigMap[:D2.fftLen/2+1,:])>0, aspect='auto', interpolation=interp)
    im.set_cmap('binary')
    

el=1
ind = np.argmin([(1-i)**2 + j**2 for i,j in zip(TP[el],FP[el])])
thresh = 10**thresh_range[ind]

ax2 = fig.add_subplot(323)
ax2.set_title('Est. tonal map $\gamma=0.5$')
ax2.set_xticklabels(())

im2 = plt.imshow(np.abs(xe[el][tonemap][:D1.fftLen/2+1,:])>thresh, aspect='auto', interpolation=interp)
im2.set_cmap('binary')

if trans:
    ax2 = fig.add_subplot(324)
    ax2.set_title('Est. trans. map $\gamma=0.5$')
    ax2.set_xticklabels(())
    im2 = plt.imshow(np.abs(xe[el][D1.N + transmap][:D2.fftLen/2+1,:])>thresh, aspect='auto', interpolation=interp)
    im2.set_cmap('binary')

el=3
ind = np.argmin([(1-i)**2 + j**2 for i,j in zip(TP[el],FP[el])])
thresh = 10**thresh_range[ind]

ax2 = fig.add_subplot(325)
ax2.set_title('Est. tonal map $\gamma=1$')

im2 = plt.imshow(np.abs(xe[el][tonemap][:D1.fftLen/2+1,:])>thresh, aspect='auto', interpolation=interp)
im2.set_cmap('binary')
if trans:
    ax2 = fig.add_subplot(326)
    ax2.set_title('Est. trans. map $\gamma=1$')
    im2 = plt.imshow(np.abs(xe[el][D1.N + transmap][:D2.fftLen/2+1,:])>thresh, aspect='auto', interpolation=interp)
    im2.set_cmap('binary')