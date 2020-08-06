"""
This script Fits S and C, that were found in the Stag fitting script

Outputs lines for S and C for extrapolation along with uncertainty

takes in a data file containing Stagnation data from existing dataset
also input liquid flyer velocity to get 
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
us_in = 16.49
use_in = 0.13

#Input data for S's and C's
steps=10000
file = 'Stag_Experiment_data_June25_2020.txt'
S=np.genfromtxt(file,dtype=float,skip_header=0,usecols=24) #S, slope
Se=np.genfromtxt(file,dtype=float,skip_header=0,usecols=25) #S, uncertainty
C=np.genfromtxt(file,dtype=float,skip_header=0,usecols=26) #S, slope
Ce=np.genfromtxt(file,dtype=float,skip_header=0,usecols=27) #S, uncertainty
SCCV1=np.genfromtxt(file,dtype=float,skip_header=0,usecols=28) #Co-Variance 1
SCCV2=np.genfromtxt(file,dtype=float,skip_header=0,usecols=29) #Co-Variance 2
SCCV3=np.genfromtxt(file,dtype=float,skip_header=0,usecols=30) #Co-Variance 3
dens = np.genfromtxt(file,dtype=float,skip_header=0,usecols=22) #liquid density
dense = np.genfromtxt(file,dtype=float,skip_header=0,usecols=23)
us = np.genfromtxt(file,dtype=float,skip_header=0,usecols=4) #liquid density
use = np.genfromtxt(file,dtype=float,skip_header=0,usecols=5)

#
def linear(x,a,b):
    return a + x*b


AS = np.zeros(steps) #S fit params
BS = np.zeros(steps)

AC = np.zeros(steps) # C fit params
BC = np.zeros(steps)

for i in range(steps):
    us_mc = us + use * sp.randn()
    S_mc = S + Se *sp.randn()
    C_mc = C + Ce * sp.randn()
    
    temp1, temp2  = curve_fit(linear, us_mc,S_mc,p0=[1,1],
                        #bounds=[[0,0,0,3000,0],[1,1,2,7000,4000]],absolute_sigma=True,max_nfev=20000)
                        absolute_sigma=True,maxfev=20000)
    AS[i] = temp1[0]
    BS[i] = temp1[1]
    

    temp1, temp2  = curve_fit(linear, us_mc,C_mc,p0=[1,1],
                        #bounds=[[0,0,0,3000,0],[1,1,2,7000,4000]],absolute_sigma=True,max_nfev=20000)
                        absolute_sigma=True,maxfev=20000)

    AC[i] = temp1[0]
    BC[i] = temp1[1]

#Collapse S params
AS_m = np.mean(AS)
AS_e = np.std(AS)
BS_m = np.mean(BS)
BS_e = np.std(BS)

#Get S covariance
temp=[]
temp.append(AS)
temp.append(BS)
S_cov = np.cov(temp)
print('S linear params')
print('A = ',AS_m,'+/-', AS_e)
print('B = ',BS_m,'+/-', BS_e)
print('S Covariance =',S_cov)
Smat=sp.linalg.cholesky(S_cov,lower=False)

#collapse C params
AC_m = np.mean(AC)
AC_e = np.std(AC)
BC_m = np.mean(BC)
BC_e = np.std(BC)

#Get C covariance
temp=[]
temp.append(AC)
temp.append(BC)
C_cov = np.cov(temp)
print('C linear params')
print('A = ',AC_m,'+/-', AC_e)
print('B = ',BC_m,'+/-', BC_e)
print('C Covariance =',C_cov)
Cmat=sp.linalg.cholesky(C_cov,lower=False)


#Grab lines with error bars
us_line = np.linspace(10,20,100)
S_line = np.zeros((steps,100))
C_line = np.zeros((steps,100))

S_mco = np.zeros(steps)
C_mco = np.zeros(steps)


for i in range(steps):
    us_in_mc = us_in + use_in * sp.randn()
    
    Sbmat = np.matmul(sp.randn(1,2),Smat)
    as1 = AS_m + Sbmat[0,0]
    bs1 = BS_m + Sbmat[0,1]
    Cbmat = np.matmul(sp.randn(1,2),Cmat)
    ac1 = AC_m + Cbmat[0,0]
    bc1 = BC_m + Cbmat[0,1]

    S_line[i,:] = linear(us_line, as1,bs1)
    C_line[i,:] = linear(us_line,ac1,bc1)

    S_mco[i] = linear(us_in_mc, as1,bs1)
    C_mco[i] = linear(us_in_mc, ac1, bc1)

    

S_f = np.mean(S_line[:,:],axis=0)
S_fe = np.std(S_line[:,:],axis=0)

C_f = np.mean(C_line[:,:],axis=0)
C_fe = np.std(C_line[:,:],axis=0)

S_out=np.mean(S_mco)
Se_out = np.std(S_mco)
C_out=np.mean(C_mco)
Ce_out = np.std(C_mco)

print('Output S and C for input us')
print('S = ',S_out,'+/-', Se_out)
print('C= ',C_out,'+/-', Ce_out)
    

#plotting
#plotting all S and C variations
#S plot
plt.figure()
plt.errorbar(dens,S,yerr = Se,xerr=dense,fmt='o', label='S, Measured',color='red')

plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.ylabel('S')
plt.xlabel('Density (kg/m^3)')
plt.savefig('ReshockSlope_Rho_fit.pdf', format='pdf', dpi=1000)
plt.figure()
#C plot
plt.errorbar(dens,C,yerr = Ce,xerr=dense,fmt='o', label='C, Measured',color='red')

plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.ylabel('C (km/s)')
plt.xlabel('Density (kg/m^3)')
plt.savefig('ReshockC0_Rho_fit.pdf', format='pdf', dpi=1000)
#Liquid Flyers instead of dens
#S plot
plt.figure()
plt.errorbar(us,S,yerr = Se,xerr=use,fmt='o', label='S, Measured',color='red')
plt.plot(us_line,S_f, color = 'blue', label='S, Fit')
plt.fill_between(us_line,S_f + S_fe, S_f - S_fe, color = 'blue', alpha = 0.3)

plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.ylabel('S')
plt.xlabel('Shock Velocity (km/s)')
plt.savefig('ReshockSlope_FlyV_fit.pdf', format='pdf', dpi=1000)

#C plot
plt.figure()
plt.errorbar(us,C,yerr = Ce,xerr=use,fmt='o', label='C, Measured',color='red')
plt.plot(us_line,C_f, color = 'blue', label='C, Fit')
plt.fill_between(us_line,C_f + C_fe, C_f - C_fe, color = 'blue', alpha = 0.3)


plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.ylabel('C (km/s)')
plt.xlabel('Shock Velocity (km/s)')
plt.savefig('ReshockC0_FlyV_fit.pdf', format='pdf', dpi=1000)
plt.show()
