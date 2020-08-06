"""

Calculating temperature profile for an isentrope decompressing from
some input shock state. Forsterite is the only implemented mat

@author: Erik Davies

This script takes in a shock state,

T_s = shock state T
T_se = uncertainty on above
us_s = shock speed
us_se = uncertainty 

and a stagnation density.

rho_st= stag state dens
rho_ste= uncertainty

Grabs the shock state of the experiment, calcs shock density, and calcs a
iseentrope from the shock state to predict temperature at a given density.


"""

import pylab as py
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.integrate as integrate
import sys
import scipy as sp
from scipy.optimize import curve_fit
from scipy import interpolate
from matplotlib import rc
from scipy.optimize import fsolve

########Plot Parameters begin############3
#These control font size in plots.
params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize':10,
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
plt.rcParams['lines.scale_dashes'] = False
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

#Inputs
T_s = 16770#shock state T
T_se = 1270#uncertainty on above
us_s = 17.76#shock state dens
us_se = 0.23#uncertainty on above
rho_st= 1939.05#stag state dens
rho_ste= 207.2#uncertainty

rho_init = 3220

#Define Functions From root et al 2018
def hugoniot(x,a,b,c,d,us):
    return a + b * (x**1) + c * (x**2) + d * (x**3) - us
#Hugoniot fit information
a1=4.63200 # Us=c+b*x+a*x^2
b1=1.45495
c1=0.00429
d1=-7.843708417433112e-04

covparamh=[[   1.831017240698122,  -0.652611265951972 ,  0.073185789070778,  -0.002609417024651],
 [ -0.652611265951972,   0.236606823014799,  -0.026846129171457,   0.000967124009405],
 [  0.073185789070778,  -0.026846129171457,   0.003083556207129,  -0.000112248130990],
 [ -0.002609417024651 ,  0.000967124009405,  -0.000112248130990,   0.000004124931939]]
lmath=sp.linalg.cholesky(covparamh,lower=False)

def gamma_fit(x,a,b,c,d,e): #fitting function for gamma values, from Davies et al. 2020
    return 2/3 + (a - 2/3)*(2597/x)**b + c*np.exp((-(x-d)**2)/(e**2))

#Gamma Paramters
A_mean=0.376568831195
B_mean=3.69951351528
C_mean=0.654310523784
D_mean=4928.94895961
E_mean=1195.83314527
#######Make MC Arrays
size = 100
steps = 10000

T_ar = sp.zeros((size,steps))
rho_ar= sp.zeros((size,steps))
gamma=sp.zeros((size,steps))

rho_shock= sp.zeros(steps)

 #################################################################     
# begin monte carlo, only one, encompasses all calculations.
j=0
while j < steps:
    us_temp = us_s + us_se * sp.randn()
    bmath=np.matmul(sp.rand(1,4), lmath) #For covariance calculation on the hugoniot
    ah=a1+bmath[0,0]
    bh=b1+bmath[0,1]
    ch=c1+bmath[0,2]
    dh=d1+bmath[0,3]

    AG1=A_mean#Gamma params, don't perturb
    BG1=B_mean
    CG1=C_mean
    DG1=D_mean
    EG1=E_mean

    rho_i = rho_init + rho_init * 0.003
    
    up=fsolve(hugoniot,5, args=(ah,bh,ch,dh,us_temp))  # grab up
    
    
    rho_shock[j] = rho_i*us_temp/(us_temp-up) #calc rho shock
    #print(rho_shock[j])
    
    T_shock = T_s + T_se * sp.randn()
    rho_stag = rho_st + rho_ste * sp.randn()
    
    rho_ar[:,j]=np.linspace(rho_stag,rho_shock[j],size) #fill density array

    #gamma function with 32% uncertainty
    gamma[:,j]=gamma_fit(rho_ar[:,j],AG1,BG1,CG1,DG1,EG1)+(gamma_fit(rho_ar[:,j],AG1,BG1,CG1,DG1,EG1)*0.32*sp.randn())
    # for values below 2597 where the fit is no longer valid
    gamma_low = gamma_fit(2597,AG1,BG1,CG1,DG1,EG1)+(gamma_fit(2597,AG1,BG1,CG1,DG1,EG1)*0.32*sp.randn())# 
        

    T_ar[-1,j] = T_shock
    for i in range(size-2,-1,-1):
        #temperautre only depends on gamma and rho ln(T) = gamma * d ln(rho)
        if rho_ar[i,j] >= 2597:
            T_ar[i,j]=np.exp(gamma[i,j] * (np.log(rho_ar[i,j]) - np.log(rho_ar[i+1,j]))
                        + np.log(T_ar[i+1,j]))
        else:
            T_ar[i,j]=np.exp(gamma_low * (np.log(rho_ar[i,j]) - np.log(rho_ar[i+1,j]))
                        + np.log(T_ar[i+1,j]))

    j =j + 1

#Collapse Data clouds
T_isen = np.median(T_ar[:,:],axis=1)
T_isene = np.std(T_ar[:,:],axis=1)

rho_isen = np.median(rho_ar[:,:],axis=1)
Trho_isene = np.std(rho_ar[:,:],axis=1)

rho_s = np.median(rho_shock)
rho_se = np.std(rho_shock)

gamma_f = np.median(gamma[:,:],axis=1)
gamma_fe = np.std(gamma[:,:],axis=1)
#Print Values
print('Shock Density =', rho_s, '+/-', rho_se, 'kg/m^3')
print('Shock Temperature =',T_s,'+/-', T_se, 'K')
print('Release Density =', rho_st, '+/-', rho_ste, 'kg/m^3')
print('Release Temperature =',T_isen[0],'+/-', T_isene[0], 'K')


#Plotting

plt.figure()
#Plot Shock and release state

plt.errorbar(rho_s,T_s,yerr=T_se,xerr=rho_se, fmt='o', color='red',label='Shock State')
plt.errorbar(rho_st,T_isen[0],yerr=T_isene[0],xerr=rho_ste,fmt='o', color='blue',label='Predicted Release State')
#Plot Quartz shock Hugoniot
plt.plot(rho_isen,T_isen,color='black',label='Calculated Isentrope')
plt.fill_between(rho_isen, T_isen-T_isene, T_isen+T_isene, alpha=0.3, color='black')



plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.xlabel('Density ($kg/m^3$)')
plt.ylabel('Temperature (1000 K)')
#plt.grid()
#plt.xlim(0,Vf + 1)
#plt.ylim(0,600)
plt.savefig('Release_Isentrope.pdf', format='pdf', dpi=1000)

plt.figure()
#Plot Shock and release state


#Plot Quartz shock Hugoniot
plt.plot(rho_isen,gamma_f,color='black',label='Davies et al. 2020')
plt.fill_between(rho_isen, gamma_f-gamma_fe, gamma_f+gamma_fe, alpha=0.3, color='black')



plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.xlabel('Density ($kg/m^3$)')
plt.ylabel('Gruneisen Parameter')
#plt.grid()
#plt.xlim(0,Vf + 1)
#plt.ylim(0,600)
plt.show()
















