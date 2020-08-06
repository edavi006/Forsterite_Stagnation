"""

Using stagnation experiment measurements to calculate initial flyer density

@author: Erik Davies

This script takes in stagnation measurements
Vf = Liquid flyer velocity (km/s)

us_w1 = window 1 shock velocity (km/s)
us_w2 = window 2 shock velocity (km/s)

For this script, w1 is quartz
w2 is tpx

Grabs the shock state of the windows based off the input shock velocity
and the applies a fitting function to a reverse impact initial flyer density
derivation. See text for details.



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

#Inputs##############################################
Vf = 17.87 #Liquid flyer velocity (km/s)
Vf_e = 0.16

us_w1 = 14.55 #window 1 shock velocity (km/s) (quartz
us_w1e = 0.0397#window 2 shock velocity (km/s)
us_w2 = 17.38#window 2 shock velocity (km/s) (tpx
us_w2e = 0.05#window 2 shock velocity (km/s)

plots = 'yes' #PRint plots? 'yes' or 'no'
plotname='stag_P_up_N3422.pdf'

####################Functions###############################

def hug_eq(x,a,b,c,d): # For hugoniots
    return a + b * (x**1) - c *(x**2) + d * (x**3)
def hug_ROOTeq(x,a,b,c,d,driv_us): #Function same as above, except equal to zero, for easy solve
    return a + b * (x**1) - c *(x**2) + d * (x**3) - driv_us
def tpx_hug(x,a,b,c,d):
    return a + b*x - c * x *np.exp(-d*x)
def tpx_root_hug(x,a,b,c,d,driv_us):
    return a + b*x - c * x *np.exp(-d*x)- driv_us

#Some Constants################################
steps=1000#Inital MC for Vales
size = 100

rho_qtzi=2651 #Window density 1 
rho_qtzei=rho_qtzi*0.003

rho_tpxi=833 #Window Density 2 
rho_tpxei=rho_tpxi*0.003

up=np.linspace(0,20,size)

#TPX Hugoniot###############################
at1=1.795
bt1=1.357
ct1=-0.694
dt1=0.273
at2=0.018
bt2=0.003
ct2=0.027
dt2=0.011

#Qtz knudson 2013##############################
a=1.754
b=1.862
c=3.364*(10**(-2))
d=5.666*(10**(-4))          
covparamh=[[2.097e-02 ,  -6.159e-03  ,5.566e-04 , -1.572e-05],
            [-6.159e-03  , 1.877e-03  ,-1.742e-04,  5.017e-06],
           [5.566e-04 , -1.742e-04  , 1.65e-05  , -4.834e-07],
           [-1.572e-05, 5.017e-06,  -4.834e-07 ,  1.441e-08]]
lmath=sp.linalg.cholesky(covparamh,lower=False)
##################################################
#Arrays

#quartz
PHmc_qtz=sp.zeros((size,steps))#mc
rhoH_qtz=sp.zeros((size,steps))#mc
PH_qtz=sp.zeros(size)#
PHe_qtz=sp.zeros(size)#
P_qtz=sp.zeros(steps)#
rho_qtz=sp.zeros(steps)#
up_qtz=sp.zeros(steps)#
Z1=sp.zeros(steps)#

#tpx
PHmc_tpx=sp.zeros((size,steps))#mc
rhoH_tpx=sp.zeros((size,steps))#mc
PH_tpx=sp.zeros(size)#
PHe_tpx=sp.zeros(size)#
P_tpx=sp.zeros(steps)#
rho_tpx=sp.zeros(steps)#
up_tpx=sp.zeros(steps)#
Z2 =sp.zeros(steps)#

#Fosterite liquid dens
rho_L = sp.zeros(steps)#
#Re-shock states
us_fo1=sp.zeros(steps)#mc
us_fo2=sp.zeros(steps)#mc
rho_fo1=sp.zeros(steps)#mc
rho_fo2=sp.zeros(steps)#mc

#Fitting arrays
S_mc=sp.zeros(steps)
C_mc=sp.zeros(steps)

print('Starting Loop')
j=0
count=0
while j in range(0,steps):
    if j == int(0.5*steps):
        print('Halfway done')
    global A_temp #So i don't have to input into a function
    global V_fly #Same as previous
    
    bmath=np.matmul(sp.rand(1,4), lmath) #For covariance calculation on the hugoniot
    aqtz=a+bmath[0,0]
    bqtz=b+bmath[0,1]
    cqtz=c+bmath[0,2]
    dqtz=d+bmath[0,3]
    
    atpx=at1+at2*sp.randn()
    btpx=bt1+bt2*sp.randn()
    ctpx=ct1+ct2*sp.randn()
    dtpx=dt1+dt2*sp.randn()
    
    rho_qtz_init=rho_qtzi+rho_qtzei*sp.randn()
    rho_tpx_init=rho_tpxi+rho_tpxei*sp.randn()

    uw_qtz=us_w1 + us_w1e*sp.randn()
    uw_tpx=us_w2 + us_w2e*sp.randn()
    V_fly=Vf + Vf_e*sp.randn()

    #QUARTZ
    #Hugoniot
    us_qtz=hug_eq(up,aqtz,bqtz,cqtz,dqtz) #getting shock velocity
    rhoH_qtz[:,j]=rho_qtz_init*us_qtz/(us_qtz-up) #getting density array
    PHmc_qtz[:,j]=rho_qtz_init*us_qtz*up # Getting pressure from rankine-hugoniot, in MPa
    PHmc_qtz[:,j]=PHmc_qtz[:,j]*(10**6) # putting pressure into Pa
    #Get the Pressure of the shock state
    #us_int = hug_ROOTeq(aqtz,bqtz,cqtz,dqtz,uw_qtz)
    up_qtz[j]=fsolve(hug_ROOTeq,5, args=(aqtz,bqtz,cqtz,dqtz,uw_qtz)) 
    rho_qtz[j]=rho_qtz_init*uw_qtz/(uw_qtz-up_qtz[j]) #getting sdhock density
    P_qtz[j]=rho_qtz_init*uw_qtz*up_qtz[j] # Getting pressure from rankine-hugoniot, in MPa
    P_qtz[j]=P_qtz[j]*(10**6) # putting pressure into Pa

    #TPX
    #Hugoniot
    us_tpx=tpx_hug(up,atpx,btpx,ctpx,dtpx) #getting shock velocity
    rhoH_tpx[:,j]=rho_tpx_init*us_tpx/(us_tpx-up) #getting density array
    PHmc_tpx[:,j]=rho_tpx_init*us_tpx*up # Getting pressure from rankine-hugoniot, in MPa
    PHmc_tpx[:,j]=PHmc_tpx[:,j]*(10**6) # putting pressure into Pa
    #Get the Pressure of the shock state
    #us_int = hug_ROOTeq(atpx,btpx,ctpx,dtpx,uw_qtz) 
    up_tpx[j]=fsolve(tpx_root_hug,5, args=(atpx,btpx,ctpx,dtpx,uw_tpx)) 
    rho_tpx[j]=rho_tpx_init*uw_tpx/(uw_tpx-up_tpx[j]) #getting shock density 
    P_tpx[j]=rho_tpx_init*uw_tpx*up_tpx[j] # Getting pressure from rankine-hugoniot, in MPa
    P_tpx[j]=P_tpx[j]*(10**6) # putting pressure into Pa

    #Now that I have those points, I can fit it.
    Z1[j] = P_qtz[j]/((V_fly - up_qtz[j]))#Quartz Impedance
    Z2[j] = P_tpx[j]/((V_fly - up_tpx[j]))#TPX Impedance
    A_temp=((Z2[j] - Z1[j])/((up_qtz[j] - up_tpx[j]))) # Combining these two for easing equation later

    #Set up fitting function

    #Setting data and inputs,
    #This fit is a P-Up fit, up is independent variable, three points, qtz, tpx, flyer vel
    P_temp = [P_qtz[j] ,P_tpx[j], 0.1] #Flyer pressure is not zero but I change it to see effect
    up_temp = [up_qtz[j] ,up_tpx[j], V_fly]
    def impedance(x,c,s): #For fitting impedances and getting Hugoniot relation.
        return (A_temp/s) *(c + s*(V_fly - x)) *(V_fly - x)
    
    temp1, temp2 = curve_fit(impedance, up_temp,P_temp, p0=[4, 1.2],bounds=[[0,0],[10,3]])

    if temp1[1] < 0:
        print('S less than zero')
    if temp1[0] < 0:
        print('C less than zero')
    
    S_mc[j]=temp1[1]
    C_mc[j]=temp1[0]


    #From this fitted function and C and S, I can solve for initial density, although I only really need S
    rho_L[j] = (A_temp / S_mc[j]) * 10**(-6) #order of magnitude to get kg/m^3

    #Grab re-shock states while we are here
    us_fo1[j] = C_mc[j] + (V_fly-up_qtz[j])*S_mc[j] #quartz
    rho_fo1[j] =rho_L[j]*us_fo1[j]/(us_fo1[j]-(V_fly-up_qtz[j]))
    us_fo2[j] = C_mc[j] + (V_fly-up_tpx[j])*S_mc[j] #tpx
    rho_fo2[j] =rho_L[j]*us_fo2[j]/(us_fo2[j]-(V_fly-up_tpx[j]))

    #The following two lines prevent super wonky values in density, if the fit simply doesn't work
    if rho_L[j] > 500 and rho_L[j] < 3000:
        j=j+1
    count=count + 1
print('Counts =',count)
print('Steps = ', steps)
#Collapse all of the data clouds for printing and plotting.

#Quartz
PH_qtz = np.median(PHmc_qtz[:,:],axis=1)*(10**-9) #and make GPa
PHe_qtz = np.std(PHmc_qtz[:,:],axis=1)*(10**-9) #and make GPa
Pss_qtz = np.median(P_qtz)*(10**-9) #and make GPa
Psse_qtz = np.std(P_qtz)*(10**-9) #and make GPa
rhoss_qtz = np.median(rho_qtz)
rhosse_qtz = np.std(rho_qtz)
upss_qtz = np.median(up_qtz)
upsse_qtz = np.std(up_qtz)
Z_qtz=np.median(Z1)
Ze_qtz=np.std(Z1)
#TPX
PH_tpx = np.median(PHmc_tpx[:,:],axis=1)*(10**-9) #and make GPa
PHe_tpx = np.std(PHmc_tpx[:,:],axis=1)*(10**-9) #and make GPa
Pss_tpx = np.median(P_tpx)*(10**-9) #and make GPa
Psse_tpx = np.std(P_tpx)*(10**-9) #and make GPa
rhoss_tpx = np.median(rho_tpx)
rhosse_tpx = np.std(rho_tpx)
upss_tpx = np.median(up_tpx)
upsse_tpx = np.std(up_tpx)
Z_tpx=np.median(Z2)
Ze_tpx=np.std(Z2)

#Forsterite params
S_f=np.median(S_mc)
S_fe=np.std(S_mc)
C_f=np.median(C_mc)
C_fe=np.std(C_mc)
rho_f=np.median(rho_L)
rho_fe=np.std(rho_L)
temp_cloud=[]
temp_cloud.append(S_mc)
temp_cloud.append(C_mc)
hug_cov = np.cov(temp_cloud)
lmat=sp.linalg.cholesky(hug_cov,lower=False)

#Fo liquid re-shock
us_fo_qtz=np.median(us_fo1)
use_fo_qtz=np.std(us_fo1)
rho_fo_qtz=np.median(rho_fo1)
rhoe_fo_qtz=np.std(rho_fo1)
us_fo_tpx=np.median(us_fo2)
use_fo_tpx=np.std(us_fo2)
rho_fo_tpx=np.median(rho_fo2)
rhoe_fo_tpx=np.std(rho_fo2)

#Print Values
#Quartz Shock state
print('==========LINE BREAK===========')
print('Quartz Shock State')
print('U_s =', us_w1, '+/-',us_w1e, 'km/s')
print('U_p =', upss_qtz, '+/-',upsse_qtz, 'km/s')
print('Density =', rhoss_qtz, '+/-',rhosse_qtz, 'kg/m^3')
print('P =', Pss_qtz, '+/-',Psse_qtz, 'GPa')
#TPX Shock state
print('==========LINE BREAK===========')
print('TPX Shock State')
print('U_s =', us_w2, '+/-',us_w2e, 'km/s')
print('U_p =', upss_tpx, '+/-',upsse_tpx, 'km/s')
print('Density =', rhoss_tpx, '+/-',rhosse_tpx, 'kg/m^3')
print('P =', Pss_tpx, '+/-',Psse_tpx, 'GPa')
#Forsterite
print('==========LINE BREAK===========')
print('Forsterite Liquid properites')
print('Liquid Density =', rho_f, '+/-',rho_fe, 'kg/m^3')
print('S =', S_f, '+/-',S_fe)
print('C =', C_f, '+/-',C_fe, 'km/s')
print('S and C covariance')
print( hug_cov)
print('==========LINE BREAK===========')
print('Fo to Qtz Reshock')
print('U_s =', us_fo_qtz, '+/-',use_fo_qtz, 'km/s')
print('Density =', rho_fo_qtz, '+/-',rhoe_fo_qtz, 'kg/m^3')
print('Fo to TPX Reshock')
print('U_s =', us_fo_tpx, '+/-',use_fo_tpx, 'km/s')
print('Density =', rho_fo_tpx, '+/-',rhoe_fo_tpx, 'kg/m^3')
print('==========LINE BREAK===========')




#Plotting
#Quick grab of fitted c and s for forsterite

Pmc_fo=sp.zeros((size,steps))#mc
P_fo=sp.zeros(size)#
Pe_fo=sp.zeros(size)#

##for i in range(steps):
##    bmat=np.matmul(sp.rand(1,2), lmat)
##    s_temp=S_f +bmat[0,0]
##    c_temp=C_f +bmat[0,1]
##    V_fly=Vf + Vf_e*sp.randn()
##    up_tpx_temp=upss_tpx + upsse_tpx * sp.randn()
##    up_qtz_temp=upss_qtz + upsse_qtz * sp.randn()
##    Zqtz_temp=Z_qtz + Ze_qtz * sp.randn()
##    Ztpx_temp=Z_tpx + Ze_tpx * sp.randn()
##    
##    A_temp=(Ztpx_temp - Zqtz_temp/(up_qtz_temp - up_tpx_temp))
##    Pmc_fo[:,j]=impedance(up,c_temp,s_temp)
for i in range(steps):
    bmat=np.matmul(sp.rand(1,2), lmat)
    s_temp=S_f +bmat[0,0]
    c_temp=C_f +bmat[0,1]
    V_fly=Vf + Vf_e*sp.randn()
    Rho_l = rho_f + rho_fe*sp.randn()
    if s_temp < 0:
        bmat=np.matmul(sp.rand(1,2), lmat)
        s_temp=S_f +bmat[0,0]
        c_temp=C_f +bmat[0,1]
    elif c_temp < 0:
        bmat=np.matmul(sp.rand(1,2), lmat)
        s_temp=S_f +bmat[0,0]
        c_temp=C_f +bmat[0,1]
    
    Pmc_fo[:,i]=(Rho_l *(c_temp + s_temp*(up))*(up))*(10**6)



P_fo = np.median(Pmc_fo[:,:],axis=1)*(10**-9) #and make GPa
Pe_fo= np.std(Pmc_fo[:,:],axis=1)*(10**-9) #and make GPa
    
#P-up

if plots == "yes":
    print('Plot requested, printing plot and saving')
    plt.figure()
    #Plot qtz the Rayleigh Line
    plt.plot([Vf,upss_qtz],[0,Pss_qtz],color='red',alpha = 0.5,label='Fo to Qtz Rayleigh Line')
    plt.errorbar(upss_qtz,Pss_qtz,yerr=Psse_qtz,xerr=upsse_qtz, color='red')
    #Plot TPX the Rayleigh Line
    plt.plot([Vf,upss_tpx],[0,Pss_tpx],color='blue',alpha = 0.5,label='Fo to TPX Rayleigh Line')
    plt.errorbar(upss_tpx,Pss_tpx,yerr=Psse_tpx,xerr=upsse_tpx, color='blue')
    #Plot Quartz shock Hugoniot
    plt.plot(up,PH_qtz,color='red',label='Qtz Hugoniot')
    plt.fill_between(up, PH_qtz-PHe_qtz, PH_qtz+PHe_qtz, alpha=0.3, color='red')
    #Plot Tpx shock Hugoniot
    plt.plot(up,PH_tpx,color='blue',label='TPX Hugoniot')
    plt.fill_between(up, PH_tpx-PHe_tpx, PH_tpx+PHe_tpx, alpha=0.3, color='blue')
    #Plot Initial Driver shock Hugoniot (not reflected)
    plt.plot(Vf-up,P_fo,color='black',label='Forsterite Liquid Hugoniot')
    plt.fill_between(Vf-up, P_fo-Pe_fo, P_fo+Pe_fo, alpha=0.3, color='black')
    #plt.fill_betweenx(P_fo, Vf-up - Vf_e, Vf-up+Vf_e,alpha=0.3, color='black')


    plt.legend(loc='upper right', fontsize='x-small',numpoints=1,scatterpoints=1)
    plt.xlabel('Particle Velocity (km/s)')
    plt.ylabel('Pressure (GPa)')
    plt.grid()
    plt.xlim(0,Vf + 1)
    plt.ylim(0,600)
    plt.savefig(plotname, format='pdf', dpi=1000)
    plt.show()
















