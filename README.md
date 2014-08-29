# GabLab

GabLab is a python module for computing sparse  time-frequency decompositions of audio with support for Gabor-based dictionaries

Example usage:

```python
import gablab as gl
import numpy as np

y = np.random.randn(10000) # test signal
scales = [256,512,1024,2048, 4096]
L = np.ceil(len(y)/float(np.max(scales)))*np.max(scales) 

# calc block boundary
y = np.hstack((y,np.zeros(L-len(y)))) # pad input to block boundary

# Build multi-scale Gabor dictionary
D = gl.DictionaryUnion(*[gl.GaborBlock(L,s) for s in scales])

# Basis pursuit denoising analysis: min ||x||_1 s.t. ||y-Dx||_2<e
x = gl.BPDN(D,y,maxerr=1e-12,maxits=20)

# Resynthesis
ye = np.real(D.dot(x))

# Error
print 'Error is: %2.5f' % np.sum((ye-y)**2)
```


