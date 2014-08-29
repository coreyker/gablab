# Denoising plots
import os,sys,glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scikits import audiolab

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('font', size=16)
matplotlib.rc('legend', fontsize=12)

# NB: change root to directory where denoising experiment was run!
root = '/Users/corey/School/Datasets/kai'
o = glob.glob(os.path.join(root,'Original','*.wav'))

# ---------------------------------------
# Calculate stats
# BPDN w/ GaborBlock(L,1024)
dirs = ['0dB_BPDN','10dB_BPDN','20dB_BPDN']
snr = np.zeros((3,6))
BPDN = np.zeros((3,2))

# Snr
for i,d in enumerate(dirs):
    f = glob.glob(os.path.join(root,d,'t_sig*.wav'))
    for j,(a,b) in enumerate(zip(o,f)):
        y = audiolab.wavread(a)[0]
        ye = audiolab.wavread(b)[0]
        r = y-ye
        snr[i][j] = 10*np.log10(y.dot(y)/r.dot(r))

# Mean
for i,el in enumerate(snr): BPDN[i][0] = np.sum(el)/6.

# Var
for i,el in enumerate(snr):
    mean = BPDN[i][0]
    BPDN[i][1] = np.sum((el - mean)**2)/5.

# 95 % confidence interval (student's t)
#BPDN[:,1] = 2.571*np.sqrt(BPDN[:,1]/6.)

# Std
BPDN[:,1] = np.sqrt(BPDN[:,1])
print BPDN



# ---------------------------------------
# Calculate SNR GBPDN1:
# GBPDN w/ GaborBlock(L,1024), Tone_factory(tonemap)
dirs = ['0dB_GBPDN','10dB_GBPDN','20dB_GBPDN']
snr = np.zeros((3,6))
GBPDN = np.zeros((3,2))

# Snr
for i,d in enumerate(dirs):
    f = glob.glob(os.path.join(root,d,'t_sig*.wav'))
    for j,(a,b) in enumerate(zip(o,f)):
        y = audiolab.wavread(a)[0]
        ye = audiolab.wavread(b)[0]
        r = y-ye
        snr[i][j] = 10*np.log10(y.dot(y)/r.dot(r))

# Mean
for i,el in enumerate(snr): GBPDN[i][0] = np.sum(el)/6.

# Var
for i,el in enumerate(snr):
    mean = GBPDN[i][0]
    GBPDN[i][1] = np.sum((el - mean)**2)/5.

# 95 % confidence interval (student's t)
#GBPDN[:,1] = 2.571*np.sqrt(GBPDN[:,1]/6.)

GBPDN[:,1] = np.sqrt(GBPDN[:,1])        
print GBPDN


# ---------------------------------------
# Calculate SNR WGL1:
# WGL-h
# Files downloaded from:
# http://homepage.univie.ac.at/monika.doerfler/StrucAudio.html
dirs = ['0dB_WGL1','10dB_WGL1','20dB_WGL1']
snr = np.zeros((3,6))
WGL1 = np.zeros((3,2))

# Snr
for i,d in enumerate(dirs):
    f = glob.glob(os.path.join(root,d,'x_sig*.wav'))
    for j,(a,b) in enumerate(zip(o,f)):
        y = audiolab.wavread(a)[0]
        ye = audiolab.wavread(b)[0]
        r = y-ye
        snr[i][j] = 10*np.log10(y.dot(y)/r.dot(r))

# Mean
for i,el in enumerate(snr): WGL1[i][0] = np.sum(el)/6.

# Var
for i,el in enumerate(snr):
    mean = WGL1[i][0]
    WGL1[i][1] = np.sum((el - mean)**2)/5.

# 95 % confidence interval (student's t)
#WGL1[:,1] = 2.571*np.sqrt(WGL1[:,1]/6.)

WGL1[:,1] = np.sqrt(WGL1[:,1])
print WGL1

# ---------------------------------------
# Calculate SNR WGL2:
# WGLS-h
# Files downloaded from:
# http://homepage.univie.ac.at/monika.doerfler/StrucAudio.html
dirs = ['0dB_WGL2','10dB_WGL2','20dB_WGL2']
snr = np.zeros((3,6))
WGL2 = np.zeros((3,2))

# Snr
for i,d in enumerate(dirs):
    f = glob.glob(os.path.join(root,d,'x_sig*.wav'))
    for j,(a,b) in enumerate(zip(o,f)):
        y = audiolab.wavread(a)[0]
        ye = audiolab.wavread(b)[0]
        r = y-ye
        snr[i][j] = 10*np.log10(y.dot(y)/r.dot(r))

# Mean
for i,el in enumerate(snr): WGL2[i][0] = np.sum(el)/6.

# Var
for i,el in enumerate(snr):
    mean = WGL2[i][0]
    WGL2[i][1] = np.sum((el - mean)**2)/5.

# 95 % confidence interval (student's t)
#WGL2[:,1] = 2.571*np.sqrt(WGL2[:,1]/6.)

WGL2[:,1] = np.sqrt(WGL2[:,1])

print WGL2


# ---------
# bar plots
# hatch: [ '/' | '\\' | '|' | '-' | '+' | 'x' | 'o' | 'O' | '.' | '*' ]
height=0.1
fig = plt.figure()
ax = fig.add_subplot(111)
pos = np.arange(3)/2.

b1 = ax.barh(pos,GBPDN[:,0],height,xerr=np.sqrt(GBPDN[:,1])**2,hatch='\\',color='0.4', error_kw=dict(elinewidth=2, ecolor='k'))
b2 = ax.barh(pos+height,BPDN[:,0],height,xerr=np.sqrt(BPDN[:,1])**2,hatch='/',color='0.55', error_kw=dict(elinewidth=2, ecolor='k'))
b3 = ax.barh(pos+2*height,WGL1[:,0],height,xerr=np.sqrt(WGL1[:,1])**2,hatch='\\',color='0.7', error_kw=dict(elinewidth=2, ecolor='k'))
b4 = ax.barh(pos+3*height,WGL2[:,0],height,xerr=np.sqrt(WGL2[:,1])**2,hatch='/',color='0.85', error_kw=dict(elinewidth=2, ecolor='k'))

plt.yticks(pos+2*height, ('0dB', '10dB', '20dB'))
ax.set_xlabel('Output SNR (dB)')
ax.set_ylabel('Input SNR (dB)')
ax.grid(True)

ax.legend([b4,b3,b2,b1],['WGLS-h','WGL-h','BPDN','G-BPDN'], loc='bottom right')
ax.set_title('Audio Denoising')
plt.show()

# outfile = 'DenoisingPlots2.eps'
# fig.savefig(outfile,pad_inches=0,bbox_inches='tight')
# os.system('pstopdf %s' % outfile)