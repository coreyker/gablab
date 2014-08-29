import os,sys,glob,pdb
import numpy as np
import scikits.audiolab as audiolab
import scikits.samplerate as samplerate
from gablab import *

'''
----------------
Processing function
_____________________
'''
def Process(z,nvar,flag):
    L = len(z)
    D = GaborBlock(L,1024,4)
    z = np.hstack((z,np.zeros(D.M-L))) # pad to block boundary
    tonemap = np.reshape(range(D.N),(D.N/D.fftLen,D.fftLen)).transpose()

    # choose objective function
    if flag=='BPDN':
        f,fgrad = BP_factory()
    elif flag=='GBPDN':
        f,fgrad = Tone_factory(tonemap,gamma=0.5)
    else: raise Exception('Unrecognized option')
        
    xe = GBPDN_momentum(D,z,f,fgrad,maxerr=nvar,maxits=500,stoptol=1e-3,muinit=1e-1,momentum=0.9,smoothinit=1e-5,anneal=0.96)
    ye = np.real(D.dot(xe))
    return ye[:L]

def fail_usage():
    print 'usage:'
    print '     python.py denoising_experiment [dir] [mode]'
    print '         [dir] is that dataset directory'
    print '         [mode] can be either BPDN or GPBDN'

def get_dataset(root):    
    http = 'http://homepage.univie.ac.at/monika.doerfler/'

    files = ['sig_1.wav', 'sig_2.wav', 'sig_3.wav', 'sig_4.wav', 'sig_5.wav', 'sig_6.wav']
    files_0dB  = ['sig_n0_1.wav', 'sig_n0_2.wav', 'sig_n0_3.wav', 'sig_n0_4.wav', 'sig_n0_5.wav', 'sig_n0_6.wav']
    files_10dB = ['sig_n10_1.wav', 'sig_n10_2.wav', 'sig_n10_3.wav', 'sig_n10_4.wav', 'sig_n10_5.wav', 'sig_n10_6.wav']
    files_20dB = ['sig_n20_1.wav', 'sig_n20_2.wav', 'sig_n20_3.wav', 'sig_n20_4.wav', 'sig_n20_5.wav', 'sig_n20_6.wav']

    if not os.path.exists(os.path.join(root,'Original')):
        os.makedirs(os.path.join(root,'Original')) 
    if not os.path.exists(os.path.join(root,'0dB')):
        os.makedirs(os.path.join(root,'0dB')) 
    if not os.path.exists(os.path.join(root,'10dB')):
        os.makedirs(os.path.join(root,'10dB')) 
    if not os.path.exists(os.path.join(root,'20dB')):
        os.makedirs(os.path.join(root,'20dB')) 

    for f in files:        
        if not os.path.exists(os.path.join(root,'Original',f)):
            os.system('wget %s%s -O %s' %(http,f,os.path.join(root,'Original',f)))
    for f in files_0dB:        
        if not os.path.exists(os.path.join(root,'0dB',f)):
            os.system('wget %s%s -O %s' %(http,f,os.path.join(root,'0dB',f)))
    for f in files_10dB:
        if not os.path.exists(os.path.join(root,'10dB',f)):
            os.system('wget %s%s -O %s' %(http,f,os.path.join(root,'10dB',f)))
    for f in files_20dB:
        if not os.path.exists(os.path.join(root,'20dB',f)):
            os.system('wget %s%s -O %s' %(http,f,os.path.join(root,'20dB',f)))

'''
------------
Main program
____________
This experiment requires data which may be downloaded from:
http://homepage.univie.ac.at/monika.doerfler/StrucAudio.html

This script will attempt to automatically download the dataset
if it is not detected (requires wget, see the get_dataset() function)

Example usage:
python denoising_experiment.py /Users/corey/School/Datasets/kai BPDN
'''

if __name__ == '__main__':
    
    if len(sys.argv) is not 3:
        fail_usage()
        exit();

    root = sys.argv[1]
    mode = sys.argv[2]        
    dirs = ['0dB','10dB','20dB'] # files for each noise-level are kept in their own directory

    get_dataset(root) # attempt to get dataset if missing

    for d in dirs:
        output_dir = os.path.join(root,d+'_'+mode)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    origFiles = glob.glob(os.path.join(root, 'Original', '*.wav'))
    
    for d in dirs:
        noisyFiles = glob.glob(os.path.join(root, d, '*.wav'))
        for a,b in zip(origFiles,noisyFiles):

            f = os.path.split(b)[1]
            print 'Processing %s' % f
            print '-----------------------------'
            y = audiolab.wavread(a)[0]
            z = audiolab.wavread(b)[0]
            r = y-z
            
            inSNR = 10*np.log10(y.dot(y)/r.dot(r))
            nvar = np.sum(np.abs(y)**2)/(10**(inSNR/10))
            
            ye = Process(z, nvar, mode)
            r = y-ye
            outSNR = 10*np.log10(y.dot(y)/r.dot(r))
        
            print 'File: %s, Input SNR = %f, output SNR = %f' % (f, inSNR, outSNR)
            audiolab.wavwrite(ye,os.path.join(root,d+'_'+mode,'t_'+f),44100.)


