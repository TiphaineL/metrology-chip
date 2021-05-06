# DW_chip

Direct-write chip - Design code

I had modified my PhD code to re-write it with function.

The code is organised as follows:
The constants of the problems and variables are in the folder problem_variables.
The optimisation is performed by minimising a cost function that takes into account the bend radius, proximity and path length difference (between waveguides).
In order to balance the contribution of the last three into the cost function, the coefficients the cost function can
be changed (ie. change the coefficients by hand, re-run optimiser until the contribution are balanced as desired. Watch out, 
there are several local minima). 
There is a cost function for the fine metrology and one for the TOF. (So the code can produce a design for both (need to run them separatly)).
There is also two different designs possible, with the fine metrology outputs going straight through the chip, or going at 90degrees. 
Before running the optimisation, the code guesses the Bezier control points, so that they are close to a solution for the optimiser to start with. 

File to run for the optimisation: problem/optimiser.py
main.py used for plotting known curves. 
