"""
This script Fits liquid density based on a sample shock velocity.

The idea here is a simple linear fit to do short range extrapolations
to predict what some other experiments might have looked like if we
actually did stagnation with them. Once again, as with the the other scripts
of this vein, uncertainty is calculated, but is expected to be large.


This takes in the files that have us-rhoL data, as well as an input
us for a predicted point based on the fit. 


"""

import matplotlib.pyplot as plt
import pylab as py
import numpy as np
import scipy as sp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit


########Plot Parameters begin############3
#These control font size in plots.
params = {'legend.fontsize': 10,
         'axes.labelsize': 12,
         'axes.titlesize':12,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
plt.rcParams.update(params)
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['axes.linewidth']= 1

plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.dashed_pattern'] = [6, 6]
plt.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
plt.rcParams['lines.dotted_pattern'] = [1, 3]
plt.rcParams['errorbar.capsize'] = 3
plt.rcParams['lines.scale_dashes'] = True
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.framealpha'] = None
plt.rcParams['legend.edgecolor'] = 'inherit'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['figure.figsize']=5,4

########Plot Parameters finish############3
#Mini input
us_in = 17.76
use_in = 0.23

#Input data for sample us and liquid density
steps=10000
file = 'Stag_Experiment_data_June25_2020.txt'
dens = np.genfromtxt(file,dtype=float,skip_header=0,usecols=22) #liquid density
dense = np.genfromtxt(file,dtype=float,skip_header=0,usecols=23)
us = np.genfromtxt(file,dtype=float,skip_header=0,usecols=4) #liquid density
use = np.genfromtxt(file,dtype=float,skip_header=0,usecols=5)

#
def linear(x,a,b):
    return a + x*b

A = np.zeros(steps) #fit params
B = np.zeros(steps)

for i in range(steps):
    us_mc = us + use * sp.randn()
    dens_mc = dens + dense *sp.randn()

    temp1, temp2  = curve_fit(linear, us_mc,dens_mc,p0=[1,1],
                        #bounds=[[0,0,0,3000,0],[1,1,2,7000,4000]],absolute_sigma=True,max_nfev=20000)
                        absolute_sigma=True,maxfev=20000)

    A[i] = temp1[0]
    B[i] = temp1[1]

#Collapse fit params
A_m = np.mean(A)
A_e = np.std(A)
B_m = np.mean(B)
B_e = np.std(B)

#Get S covariance
temp=[]
temp.append(A)
temp.append(B)
S_cov = np.cov(temp)
print('linear params')
print('A = ',A_m,'+/-', A_e)
print('B = ',B_m,'+/-', B_e)
print('Covariance =',S_cov)
Smat=sp.linalg.cholesky(S_cov,lower=False)


#Grab line with error bars
us_line = np.linspace(10,20,100)
dens_line = np.zeros((steps,100))

dens_mco = np.zeros(steps)

for i in range(steps):
    us_in_mc = us_in + use_in * sp.randn()
    
    Sbmat = np.matmul(sp.randn(1,2),Smat)
    as1 = A_m + Sbmat[0,0]
    bs1 = B_m + Sbmat[0,1]

    dens_line[i,:] = linear(us_line, as1,bs1) #makeing the line

    dens_mco[i] = linear(us_in_mc, as1,bs1)

dens_f = np.mean(dens_line[:,:],axis=0)
dens_fe = np.std(dens_line[:,:],axis=0)

dens_out=np.mean(dens_mco)
dense_out = np.std(dens_mco)

print('Output liquid density for input us')
print('rho_L = ',dens_out,'+/-', dense_out, 'kg/m^3')

#PLotting###################################################
#S plot
plt.figure()
plt.errorbar(us,dens,yerr = dense,xerr=use,fmt='o', label='Liquid Density, Measured',color='red')
plt.plot(us_line,dens_f, color = 'blue', label='Density Fit')
plt.fill_between(us_line,dens_f + dens_fe, dens_f - dens_fe, color = 'blue', alpha = 0.3)

plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.ylabel('Liquid Density')
plt.xlabel('Shock Velocity (km/s)')
plt.savefig('SampleUSvLdens_fit.pdf', format='pdf', dpi=1000)

plt.show()






