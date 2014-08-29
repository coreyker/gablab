# GabLab is a Python library for sparse approximation using Gabor dictionaries
# GabLab was written by Corey Kereliuk
# GabLab is released under a CC-BY-NC-SA license:
# http://creativecommons.org/licenses/by-nc-sa/3.0/

import pdb
import numpy as np
import scikits.audiolab as audiolab
import scikits.samplerate as samplerate
from matplotlib import rc, pyplot

def odd(n): return n+(1-n%2)
def even(n): return n+n%2
'''
______________________________________________
Class implementing a Gabor dictionary
Features: fast inner products using the FFT
______________________________________________
'''
class GaborBlock:
    def __init__(self,L,winLen=1024, timeOver=2, freqOver=1):
        self.transposed = False

        self.timeOver = timeOver
        self.freqOver = freqOver
        self.winLen = even(int(winLen))
        self.stride = int(self.winLen/float(self.timeOver))
        self.fftLen = int(2**np.ceil(np.log2(self.winLen*self.freqOver)))

        self.window = np.hanning(odd(self.winLen))[1:]
        #self.window = np.sqrt(self.window)
        self.window /= np.linalg.norm(self.window)        
        self.winSq = self.window ** 2
        
        self.winSup = np.arange(0,self.winLen,dtype=int)

        self.M = int(np.ceil(L/float(self.stride))*self.stride)
        self.N = int(np.ceil(L/float(self.stride))*self.fftLen)
        self.shape = (self.M, self.N)
        
        # calc diagonal frame operator
        self.diag = np.zeros(self.M, dtype=float)
        
        for t in range(0,self.M,self.stride):
            sup = (t + self.winSup) % self.M
            self.diag[sup] += self.winSq

        self.diag =  np.sqrt(self.fftLen * self.diag)

        # tight frame windows
        self.winmat = np.zeros((self.M/self.stride,self.winLen))
        for i,t in enumerate(range(0,self.M,self.stride)):
            sup = (t + self.winSup) % self.M
            self.winmat[i,:] = self.window / self.diag[sup]
        
    def transpose(self):
        tmpSelf = GaborBlock(self.M,self.winLen,self.timeOver,self.freqOver)
        tmpSelf.transposed = not self.transposed
        return tmpSelf

    def conj(self):
        return self
    
    def diagonal(self):
        return self.diag
    
    def  dot(self, vec):
        if self.transposed:
            assert vec.size == self.M, 'Incompatible dimensions. Consider padding the input vector with zeros to match the dictionary size.'
            return self.analyze(vec)
        else:
            assert vec.size == self.N, 'Incompatible dimensions.'
            return self.synthesize(vec)

    def analyze(self,vec):
        sup = np.tile(np.arange(0,self.M,self.stride), (self.winLen,1)).transpose()
        sup += np.tile(self.winSup,(self.M/self.stride,1))          
        sup %=self.M
                
        x = np.fft.fft(vec[sup] * self.winmat, self.fftLen)
        x = np.reshape(x,self.N)
        return x

    def synthesize(self,vec):
        S = np.fft.ifft(np.reshape(vec, (self.M/self.stride,self.fftLen))) * self.fftLen
                        
        # OLA
        y = np.zeros(self.M, dtype=complex)
                 
        for i,t in enumerate(range(0,self.M,self.stride)):
            y[(t + self.winSup) % self.M] += S[i,:self.winLen] * self.winmat[i,:]
        #y /= self.diagonal()
        return y
                        
#    def __add__(self,vector):
#    def concatenate()

'''
________________________________________________________________
Class allowing one to combine several GaborBlocks together
________________________________________________________________
'''
class DictionaryUnion:
    def __init__(self,*blocks):
        self.transposed = False
        self.blockList = [b for b in blocks]

        self.M = self.blockList[0].M
        self.N = 0
        for b in self.blockList:
            assert b.M == self.M, 'Incompatible dimensions.'
            self.N += b.N
        self.shape = (self.M, self.N)
        self.nrm = np.sqrt(len(self.blockList))
        
    def transpose(self):
        tmpSelf = DictionaryUnion(*[b.transpose() for b in self.blockList])
        tmpSelf.transposed = not self.transposed
        return tmpSelf

    def conj(self):
        return self
        
    def dot(self, vec):
        if self.transposed:
            #assert vec.size == self.M, 'Incompatible dimensions. Consider padding the input vector with zeros to match the dictionary size.'
            return self.analyze(vec)
        else:
            #assert np.array(vec).size == self.N, 'Incompatible dimensions.'
            return self.synthesize(vec)

    def analyze(self,vec):        
        x = np.hstack([b.analyze(vec) for b in self.blockList])
        x /= self.nrm
        #x = [b.analyze(vec)/self.nrm for b in self.blockList]
        return x
                 
    def synthesize(self,vec):
        y = 0
        offset = 0
        for i,b in enumerate(self.blockList):
            y += b.synthesize(vec[offset:offset+b.N])
            #y += b.synthesize(vec[i])
            offset += b.N
        y /= self.nrm
        return y

    def __getitem__(self,key):
        return self.blockList[key]

    def __len__(self):
        return len(self.blockList)
    
    def parts(self,vec):
        i  = 0
        result=[]
        for b in self.blockList:
            part = vec[i:i+b.N]
            sup = np.reshape(range(0,b.N),(b.N/b.fftLen,b.fftLen))           
            result.append(part[sup])
            i+=b.N
        return tuple(result)
            
'''
____________________________________________________________
Basis pursuit denoising (BPDN) 

Solves the problem:
    min ||x||_1 s.t. ||b-Ax||_2<e
given A and b.

Features: should work with these dictionary types:
               GaborBlock/DictionaryUnion/NumPy Matrix

Implemented using fast projected gradient descent
____________________________________________________________
'''
def BPDN(A,b,maxerr=1e-12,maxits=1000,stoptol=1e-3):
    # require b == A.dot(A.transpose().dot(b))
    
    # init
    mu = 1e-1
    x = xi = A.conj().transpose().dot(b)
    r = np.abs(x)
    theta = np.angle(x)             
    obj = np.linalg.norm(x,1) # objective func. value
    t = 1
    
    it=0
    while it<maxits:
        it+=1

        # gradient step
        grad = 1
        step = r - mu*grad
        z = (step>0)*step
        u = z * np.exp(1j * theta)

        # projection
        Au = A.dot(u)
        lamb = max(0, (1/np.sqrt(maxerr)) * np.linalg.norm(b-Au) - 1)
        xlast = x
        x = u + (lamb/(1+lamb)) * (xi - A.conj().transpose().dot(Au))

        # check
        if  np.linalg.norm(x,1) > obj:
            if mu<stoptol:
                return xlast
            else:
                print 'reducing step size'
                mu /= 2 # reduce step size
                #break
                continue
        else:
            if obj-np.linalg.norm(x,1)<stoptol:
                return x
            else:
                obj = np.linalg.norm(x,1)
                
                # FISTA update
                tlast = t
                t = (1 + np.sqrt(1 + 4*t**2))/2;
                y = x + (x - xlast)*(tlast-1)/t;
        
                r = np.abs(y)
                theta = np.angle(y)
                        
        print '%d: ||x||_1 = %f' % (it, obj)

    return x

'''
_______________________________________________________________
Generalized basis pursuit denoising (G-BPDN)

Solves the problem:
    min ||f(|x|)||_1 s.t. ||b-Ax||_2<e
given A and b and f().  Ideally the function f() will sparsify |x|

Features: should work with these dictionary types:
               GaborBlock/DictionaryUnion/NumPy Matrix

Implemented using fast projected gradient descent
_______________________________________________________________
'''
def GBPDN_fista(A,b,f,fgrad,maxerr=1e-12,maxits=1000,stoptol=1e-4,smooth=1e-4,decay=1):
    # require b == A.dot(A.transpose().dot(b))
    
    # init
    mu = 1e-3#smooth #1e-1*smooth
    #if stoptol>smooth: stoptol=smooth
    
    x = A.conj().transpose().dot(b)
    r = np.abs(x)    
    theta = np.angle(x)             

    xi = np.copy(x)
    y  = np.copy(x)

    obj = np.inf#np.linalg.norm(f(r),1) # objective func. value
    t = 1
    
    it=0
    while it<maxits:
        it+=1
        smooth *= decay
        
        # gradient step
        # *** when deleting comments get rid of refs to theta too (not needed anymore)
        #grad = fgrad(PInf(f(r)/smooth))
        #step = r - mu*grad
        #z = (step>0)*step
        #u = z * np.exp(1j * theta)

        #grad = np.exp(1j * theta) * fgrad(PInf(f(r)/smooth))
        grad = PInf(y/smooth) * fgrad(PInf(f(r)/smooth))
        u = y-mu*grad
        
        # projection
        Au = A.dot(u)
        lamb = max(0, (1/np.sqrt(maxerr)) * np.linalg.norm(b-Au) - 1)
        xlast = np.copy(x)
        x = u + (lamb/(1+lamb)) * (xi - A.conj().transpose().dot(Au))
        
        # check
        if  np.linalg.norm(f(np.abs(x)),1) > obj:
            if 0:#mu<stoptol:
                return xlast
            else:
                print 'reducing step size'
                mu /= 2 # reduce step size
                y = np.copy(xlast)
                r = np.abs(y)
                theta = np.angle(y)
                #break
                continue
        else:
            if 0:#obj-np.linalg.norm(f(np.abs(x)),1)<stoptol:
                print 'test'
                return x
            else:
                obj = np.linalg.norm(f(np.abs(x)),1)
                
                # FISTA update
                tlast = np.copy(t)
                t = (1 + np.sqrt(1 + 4*t**2))/2;
                y = x + (x - xlast)*(tlast-1)/t;
        
                r = np.abs(y)
                theta = np.angle(y)
                        
        print '%d: obj = %f' % (it, obj)

    return x

def GBPDN_momentum(A, b, f, fgrad, maxerr=1e-12, maxits=1000, stoptol=1e-4, muinit=1e-2, smoothinit=1e-4, momentum=0.9, anneal=0.98):
    # require b == A.dot(A.transpose().dot(b))
    
    # init
    mu   = muinit
    smooth = smoothinit
    beta = momentum

    x0   = A.conj().transpose().dot(b)
    x    = np.copy(x0)
    grad = 0    
    prev_obj  = np.inf

    it=0
    while it<maxits:
        it+=1
        mu *= anneal
        smooth *= anneal
        
        # gradient step w/ momentum
        grad = beta*grad - mu*(1-beta)*PInf(x/smooth)*fgrad(PInf(f(np.abs(x))/smooth))
        u = x + grad
        
        # projection
        Au    = A.dot(u)
        lamb  = max(0, (1/np.sqrt(maxerr)) * np.linalg.norm(b-Au) - 1)
        xlast = np.copy(x)
        x     = u + (lamb/(1+lamb)) * (x0 - A.conj().transpose().dot(Au))
        
        # check
        obj = np.linalg.norm(f(np.abs(x)),1)
        if obj > prev_obj:
            #x = np.copy(xlast)
            print 'obj increased'
            #break
        elif np.abs(obj - prev_obj)<stoptol:
            print 'stoptol reached'
            break
        else:                    
            prev_obj = np.copy(obj)
        print '%d: obj = %f' % (it, obj)        

    return x

'''
_______________________________________________________________
Some helper functions
_______________________________________________________________
'''
# ---------------------------
# Projection onto the inf-norm ball
def PInf(x): return np.exp(1j*np.angle(x)) * np.clip(np.abs(x),0,1)

# -----------------------------
# Basis pursuit objective and gradient
def BP_factory():
    def f(x): return x
    def fgrad(x): return x
    return f,fgrad

# ----------------------------
# Fused lasso objective and gradient
def FL_factory(gamma=0.5):
    def f(x):
        d = np.concatenate(([x[0]], np.diff(x)))
        return np.concatenate(((1-gamma)*d, gamma*x))
        
    def fgrad(x):
        xa = x[:len(x)/2]
        xb = x[len(x)/2:]
        d = np.concatenate((-np.diff(xa),[xa[-1]]))
        return (1-gamma)*d + gamma*xb

    return f,fgrad

# --------------------------------------
# Tonal/transient adjacency objective and gradient
def TT_factory(tonemap, transmap, gamma1=0.5, gamma2=0.5, zeta=0.5):
    def f(x):
        xtone = x[tonemap]
        xtrans = x[tonemap.size+transmap].transpose()

        #toneDiff = np.concatenate((np.diff(xtone).transpose().flatten(),-xtone[:,-1]))
        #transDiff = np.concatenate((np.diff(xtrans).flatten(),-xtrans[:,-1]))

        #toneDiff = np.concatenate((xtone[:,0], np.diff(xtone).transpose().flatten()))
        #transDiff = np.concatenate((xtrans[:,0], np.diff(xtrans).flatten()))

        toneDiff = np.concatenate((xtone[:,0].reshape((xtone.shape[0],1)), np.diff(xtone)),1).transpose().flatten()
        transDiff = np.concatenate((xtrans[:,0].reshape((xtrans.shape[0],1)), np.diff(xtrans)),1).flatten()
        
        tonalObj = np.concatenate(((1-gamma1)*toneDiff, gamma1*xtone.transpose().flatten()))
        transObj = np.concatenate(((1-gamma2)*transDiff, gamma2*xtrans.flatten()))

        #pdb.set_trace()
        return np.concatenate((zeta*tonalObj, (1-zeta)*transObj))
        #return np.concatenate((tonalObj, transObj))

    def fgrad(x):
        # unpack x
        xa = x[:tonemap.size]
        xb = x[tonemap.size:2*tonemap.size]
        xc = x[2*tonemap.size:2*tonemap.size+transmap.size]
        xd = x[2*tonemap.size+transmap.size:]
        
        xtone = xa[tonemap]
        xtrans = xc[transmap].transpose()

        #toneDiff = np.concatenate((-xtone[:,0], -np.diff(xtone).transpose().flatten()))
        #transDiff = np.concatenate((-xtrans[:,0], -np.diff(xtrans).flatten()))

        #toneDiff = np.concatenate((-np.diff(xtone).transpose().flatten(),xtone[:,-1]))
        #transDiff = np.concatenate((-np.diff(xtrans).flatten(),xtrans[:,-1]))

        toneDiff = np.concatenate((-np.diff(xtone), xtone[:,-1].reshape((xtone.shape[0],1))),1).transpose().flatten()
        transDiff = np.concatenate((-np.diff(xtrans), xtrans[:,-1].reshape((xtrans.shape[0],1))),1).flatten()
        
        #pdb.set_trace()
        return np.concatenate((zeta*((1-gamma1)*toneDiff + gamma1*xb), (1-zeta)*((1-gamma2)*transDiff + gamma2*xd)))
        #return np.concatenate((((1-gamma1)*toneDiff + gamma1*xb), ((1-gamma2)*transDiff + gamma2*xd)))
    return f,fgrad

# -------------------------------
# Tonal adjacency objective and gradient
def Tone_factory(tonemap, gamma=0.5):   
    def f(x):
        xtone = x[tonemap]
        toneDiff = np.concatenate((xtone[:,0].reshape((xtone.shape[0],1)), np.diff(xtone)),1).transpose().flatten()
        return np.concatenate(((1-gamma)*toneDiff, gamma*x))

    def fgrad(x):
        # unpack x
        xa = x[:tonemap.size]
        xb = x[tonemap.size:2*tonemap.size]
        xtone = xa[tonemap]

        toneDiff = np.concatenate((-np.diff(xtone), xtone[:,-1].reshape((xtone.shape[0],1))),1).transpose().flatten()

        return (1-gamma)*toneDiff + gamma*xb
    return f,fgrad


'''
_____________
Test scripts
_____________
'''
def TestGaborBlock():
    # _____________________________________
    print 'Test: GaborBlock dictionary'
    L = 44100
    b = np.random.random(L) # random test vector
    A = GaborBlock(L, 2048, 4, 2) # dictionary
    b = np.hstack((b,np.zeros(A.M-L))) # pad to block boundary

    # Analysis
    x = A.conj().transpose().dot(b)
    # Synthesis
    be = np.real(A.dot(x)) 
    # Error
    print 'Error (should be zero): %1.12f' % np.sum((b-be)**2)
    print '----------------------------------------'
    
def TestDictionaryUnion():
    # _____________________________________
    print 'Test: Dictionary union'
    L = 44100
    b = np.random.random(L)
    A = GaborBlock(L, 2048, 4) # dictionary    
    B = GaborBlock(A.M, 512) # dictionary
    C = DictionaryUnion(A,B)
    b = np.hstack((b,np.zeros(C.M-L)))

    # Analysis
    x = C.conj().transpose().dot(b)
    
    # Synthesis
    be = np.real(C.dot(x))
    # Error
    print 'Error (should be zero): %1.12f' % np.sum((b-be)**2)
    print '----------------------------------------'

def TestBPDN1():
    # ________________________________________
    print 'Test: basis pursuit decomposition'

    L = 44100
    b = np.random.random(L)
    A = GaborBlock(L, 2048, 4) # dictionary    
    B = GaborBlock(A.M, 512) # dictionary
    C = DictionaryUnion(A,B)
    b = np.hstack((b,np.zeros(C.M-L)))
    
    e = 1e-1
    x = BPDN(C,b,e,10)
    be = np.real(C.dot(x))
    print 'Error (should be <= %1.12f): %1.12f' % (e,np.sum((b-be)**2))
    print '----------------------------------------'
    
def TestBPDN2():
    # ________________________________________
    print 'Test: basis pursuit decomposition'

    fs = 8000
    btmp = audiolab.wavread('glockenspiel.wav')[0]
    b = samplerate.resample(btmp, fs/44100.,'sinc_best')
    L = len(b)

    A = GaborBlock(L,1024)
    B = GaborBlock(A.M,64)
    C = DictionaryUnion(A,B)
    b = np.hstack((b,np.zeros(C.M-L)))
    
    e = 1e-2
    x = BPDN(C,b,e,100)
    ye = np.real(C.dot(x))
    print 'Error (should be <= %f): %f' % (e,np.sum((b-ye)**2))
    print '----------------------------------------'
    
    xtone = x[:A.N]
    xtrans = x[A.N:]
    ytone = np.real(A.dot(xtone))
    ytrans = np.real(B.dot(xtrans))

    audiolab.wavwrite(ytone,'ytone.wav',fs)
    audiolab.wavwrite(ytrans,'ytrans.wav',fs)

    # tonal decomp
    m = np.log10(np.abs(A.conj().transpose().dot(ytone)))
    tfgrid = np.reshape(range(0,A.N),(A.N/A.fftLen,A.fftLen))
    tfgrid = tfgrid[:,:A.fftLen/2+1]

    pyplot.subplot(2,1,1)
    pyplot.imshow(m[tfgrid].transpose(), aspect='auto', interpolation='bilinear', origin='lower')

    # transient decomp
    m = np.log10(np.abs(B.conj().transpose().dot(ytrans)))
    tfgrid = np.reshape(range(0,B.N),(B.N/B.fftLen,B.fftLen))
    tfgrid = tfgrid[:,:B.fftLen/2+1]

    pyplot.subplot(2,1,2)
    pyplot.imshow(m[tfgrid].transpose(), aspect='auto', interpolation='bilinear', origin='lower')

    pyplot.show()

def TestGBPDN1():   
    N=200
    M=50

    # tight frame
    A = np.fft.fft(np.eye(N))/np.sqrt(N)
    A = A[:M,:]
    
    # support
    L = 20
    seeds = np.random.randint(0,N-L,4)
    #seeds = [20, 50, 100]
    x = np.zeros(N)
    
    for n in seeds:
        x[n:n+L] = 0.25 + np.random.rand(1)

    # synth
    b = A.dot(x)        
    nvar = 1e-12
       
    print 'Running basis pursuit denoising'    
    f,fgrad = FL_factory(gamma=1)        
    xe_bpdn = GBPDN_momentum(A,b,f,fgrad,maxerr=nvar,maxits=5000,stoptol=1e-5,muinit=1e-1,smoothinit=1e-4,momentum=0.9,anneal=0.96)
    
    print 'Running generalized basis pursuit denoising'        
    f,fgrad = FL_factory(gamma=0.75)
    xe_gbpdn = GBPDN_momentum(A,b,f,fgrad,maxerr=nvar,maxits=5000,stoptol=1e-5,muinit=1e-1,smoothinit=1e-4,momentum=0.9,anneal=0.96)
    
    rc('text', usetex=True)
    rc('font', family='serif')    
    
    pyplot.figure()
    
    pyplot.plot(x, color='k', linewidth=4, alpha=0.4)
    pyplot.plot(np.real(xe_bpdn), color='k')
    pyplot.plot(np.real(xe_gbpdn), color='r')
    pyplot.grid(True)            
    pyplot.xlabel('Dictionary index')
    pyplot.ylabel('Coefficient amplitude')
    pyplot.legend(('True', 'BPDN', 'GBPDN'))
    pyplot.show()

def TestGBPDN2():    
    # ________________________________________
    print 'Test: generalized basis pursuit decomposition'

    fs = 8000.
    btmp,fstmp,fmt = audiolab.wavread('glockenspiel.wav')
    b = samplerate.resample(btmp, fs/fstmp,'sinc_best')
    L = len(b)

    A = GaborBlock(L,1024)
    B = GaborBlock(A.M,64)
    C = DictionaryUnion(A,B)
    b = np.hstack((b,np.zeros(C.M-L))) # pad to block boundary
    spow = 10*np.log10(b.conj().dot(b))
    
    # additive white noise
    snr = 15
    nvar = np.sum(np.abs(b)**2)/(10**(snr/10)) # 1e-2    
    n = np.sqrt(nvar)*np.random.randn(C.M)/np.sqrt(C.M)
    
    tonemap = np.reshape(range(A.N),(A.N/A.fftLen,A.fftLen)).transpose()
    transmap = np.reshape(range(B.N),(B.N/B.fftLen,B.fftLen)).transpose()
    
    f,fgrad = BP_factory()
    #f,fgrad = TT_factory(tonemap,transmap)
        
    xe = GBPDN_momentum(C,b+n,f,fgrad,maxerr=nvar,maxits=200,stoptol=1e-3,muinit=1e-1,momentum=0.9,smoothinit=1e-5,anneal=0.96)    
    ye = np.real(C.dot(xe))
    r = b-ye;
    rpow = 10*np.log10(r.conj().dot(r))
                       
    print 'SNR = %f' % (spow-rpow)

    ynoise = np.array(samplerate.resample(b+n,fstmp/fs,'sinc_best'),dtype='float64')
    ydenoise = np.array(samplerate.resample(ye,fstmp/fs,'sinc_best'),dtype='float64')
    
    yetone = np.real(A.dot(xe[:tonemap.size]))
    yetone = np.array(samplerate.resample(yetone,fstmp/fs,'sinc_best'),dtype='float64')

    yetrans = np.real(B.dot(xe[tonemap.size:]))
    yetrans = np.array(samplerate.resample(yetrans,fstmp/fs,'sinc_best'),dtype='float64')
    
                      
    print 'Error (should be <= %f): %f' % (nvar,np.sum((r)**2))
    print '----------------------------------------'
    
    ## s = [512,2048] 
    ## L = 10.*s[1]
    ## b = np.arange(L)/L
    ## A = GaborBlock(L, s[1]) # dictionary    
    ## B = GaborBlock(L, s[0]) # dictionary
    ## C = DictionaryUnion(A,B)
    
    ## #tonemap = np.reshape(range(A.N),(A.fftLen,A.N/A.fftLen))
    ## #transmap = np.reshape(range(B.N),(B.fftLen,B.N/B.fftLen))
    ## tonemap = np.reshape(range(A.N),(A.N/A.fftLen,A.fftLen)).transpose()
    ## transmap = np.reshape(range(B.N),(B.N/B.fftLen,B.fftLen)).transpose()
    
    ## e = 1e-1

    ## #f,fgrad = BP_factory()
    ## #f,fgrad = TT_factory(tonemap,transmap,gamma1=1,gamma2=1,zeta=0.5)
    ## f,fgrad = TT_factory(tonemap,transmap)
    
    ## x = GBPDN(C,b,f,fgrad,e,20)
    
    ## be = np.real(C.dot(x))
    ## #be = C.dot(x)
    ## pyplot.plot(np.abs(x) - np.abs(C.transpose().dot(b)))
    ## pyplot.show()
    ## print 'Error (should be < %1.12f): %1.12f' % (e,np.linalg.norm(b-be)**2)
    ## print '----------------------------------------'
    
if __name__ == '__main__':
    #TestGaborBlock()
    #TestDictionaryUnion()
    #TestBPDN1()
    #TestBPDN2()
    TestGBPDN1()
    #TestGBPDN2()
