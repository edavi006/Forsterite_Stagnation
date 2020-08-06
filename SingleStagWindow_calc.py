"""
This script Takes in a S and C and calculates a liquid density for
a given liquid flyer velocity and window velocity.

This script is written specifically for Z3172, where only one window
is available for stagnation. However, it is general such that any
S, C, and liquid velocity may be input.

It is noted here that a quartz Hugoniot is set up currently, so
if other 

Inputs
S = Hugoniot slope
Se = uncertainty on above
C = Hugoniot sound speed
Ce = uncertainty on above

Fly_v = liquid flyer velocity
fly_ve = uncertainty on above

us_w = window shock velocity
use_w = uncertainty on above

A note. Experimental densities can be recreated (close anyways)
from the initial calculation by inputing the covariance matrix that
was derived from the original calculation of liquid densities.
This was proof of concept that it works. Without a covariance
matrix however, this does not provide a very workable liquid density.
It still has issues, and predicting anything with S and C should not
be taken as data here. Uncertainties on those values are simply to
large for anything to really be done.


"""

import matplotlib.pyplot as plt
import pylab as py
import numpy as np
import scipy as sp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.optimize import fsolve


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

########Plot Parameters finish################################

#Input
S = 1.03113856221975#Hugoniot slope
Se = 0.358471672268493#uncertainty on above
C = 4.85363596770875#Hugoniot sound speed
Ce = 1.219752321978#uncertainty on above

#SC_cov = [[ 0.18870064, -0.52872613], #Covariance matrix to recreate experimental density
#     [-0.52872613,  1.55844094]]
#Smat =sp.linalg.cholesky(SC_cov,lower=False)

Fly_v = 18.4#liquid flyer velocity
Fly_ve = 0.27#uncertainty on above

us_w = 14.9#window shock velocity
use_w = 0.1#uncertainty on above
#other constants
steps = 10000

###############################################################
#Quartz Hug setup
def hug_eq(x,a,b,c,d): # For hugoniots
    return a + b * (x**1) - c *(x**2) + d * (x**3)
def hug_ROOTeq(x,a,b,c,d,driv_us): #Function same as above, except equal to zero, for easy solve
    return a + b * (x**1) - c *(x**2) + d * (x**3) - driv_us
#Qtz knudson 2013
a=1.754
b=1.862
c=3.364*(10**(-2))
d=5.666*(10**(-4))          
covparamh=[[2.097e-02 ,  -6.159e-03  ,5.566e-04 , -1.572e-05],
            [-6.159e-03  , 1.877e-03  ,-1.742e-04,  5.017e-06],
           [5.566e-04 , -1.742e-04  , 1.65e-05  , -4.834e-07],
           [-1.572e-05, 5.017e-06,  -4.834e-07 ,  1.441e-08]]
lmath=sp.linalg.cholesky(covparamh,lower=False)
rho_qtzi=2651 #Window density 1 
rho_qtzei=rho_qtzi*0.003
##################################################
#Quick summary of what I am doing
#For a reverse impact, of liquid flyer into a known window
#P-P0L = rho_0 * UsL * (V_fly - up)
# P0L -> 0, P and up are obtained from impedance match with quartz
# which is why i need the above hugoniot. UsL is linear for this work
# and fit using a seperate script. This is all done in a MC loop
# for quick and easy uncertainty propagation
#########################Arrays#######################
P_qtz=sp.zeros(steps)#
#rho_qtz=sp.zeros(steps)#
up_qtz=sp.zeros(steps)#

rhoL_mc = sp.zeros(steps)#

for i in range(steps):
    #perturb things
    bmath=np.matmul(sp.rand(1,4), lmath) #For covariance calculation on the hugoniot
    aqtz=a+bmath[0,0]
    bqtz=b+bmath[0,1]
    cqtz=c+bmath[0,2]
    dqtz=d+bmath[0,3]    

    us_mc = us_w + use_w * sp.randn()
#    smath=np.matmul(sp.rand(1,2), Smat) #For covariance calculation on the hugoniot
#    S_mc = S+smath[0,0]
#    C_mc = C+smath[0,1]
    S_mc = S+Se * sp.randn()
    C_mc = C+Ce * sp.randn()
    fly_mc = Fly_v + Fly_ve * sp.randn()
    rho_qtz_init=rho_qtzi+rho_qtzei*sp.randn()

    up_qtz[i] = fsolve(hug_ROOTeq,5, args=(aqtz,bqtz,cqtz,dqtz,us_mc)) 
    P_qtz[i]=rho_qtz_init*us_mc*up_qtz[i] # Getting pressure from rankine-hugoniot, in MPa
    P_qtz[i]=P_qtz[i]*(10**6) # putting pressure into Pa
    
    #Calc the liquid density now that I have everything
    
    rhoL_mc[i] = P_qtz[i]*(10**(-9)) / ( (C_mc + S_mc * (fly_mc - up_qtz[i]))*(fly_mc - up_qtz[i]))
    

#Grab the mean and std
rhoL=np.median(rhoL_mc)
rho_Le = np.std(rhoL_mc)

print('Liquid Density =', rhoL*1000, '+/-', rho_Le*1000, 'kg/m^3')














