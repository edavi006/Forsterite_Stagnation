# Forsterite_Stagnation
 Repository of all data and calculations required for the Stagnation paper

These collections of scripts re-create the plots and calculations found in 
"Probing the Forsterite L-V Dome: Temperature and Density at the Phase boundary" Davies et al. 2020

Traditional Python scripts are used to analyze the majority of data here.

"Stag_FittingForDensity.py" fits the reverse impact data, fitting S and C_0, then calculating the density of 
the flyer as described in the main text. Time to run is on order 10s of seconds.

"Calc_releaseIsentrope.py" Grabs the shock state of the experiment, calculates shock density, and calculates an
isentrope from the shock state to predict temperature at a given density. This is a useful script to probe
what the L-V dome temperature might be.

The python notebook "L-V_Dome_DepressedTCalc.ipynb" calculates the apparant temperature of the liquid-vapor 
phase boundary, based on optical depth calculations described in the main text.

"FittingS_and_C.py" Fits S and C_0 to try and determine linear trends. The degree of uncertainty here does
not allow for meaningful interpretation.

"SingleStagWindpw_calc.py" takes in a S and C_0 for single window reverse impact experiments, and calculates
density of the flyer. This was not used in this work, but was developed during the process to attempt to 
calculate a data set that was un-used. Less uncertain S and C_0 would make this a viable calculation.

"UsDensFit_and_output.py" This script fits the reverse impact density data and extrapolates linearly to predict
density for nearby experiments. This is not a robust calculation, mostly used to help predict release temepratures.

For information about the EOS used in this work see separate documentation in 
"Stewart, S. T., Davies, E. J., Duncan, M. S., Lock, S. J., Root, S., Townsend, J. P., ... & Jacobsen, S. B. (2019). 
The shock physics of giant impacts: Key requirements for the equations of state. arXiv preprint arXiv:1910.04687."
for an in depth description of the EOS used in this work.