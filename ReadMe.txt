--------------------------------------------------
Read Me
--------------------------------------------------
This repository contains 2 python files. 
> Functions.py - contains all the functions
> Execution.py - this is used to call up individual functions from functions.py

The most important variable in most functions is "Dataset", which refers to the used Dataset. 
This can either be "BasicRadial", "C05", "C06", "C07", "Cut" or "Delta_Catenin". 
"Cut" and "Delta_Catenin" rely on individual cell tracking, the rest has given the an interpolated velocity field (and in case of "C05" and "C06" a cell density)

______________________________________________________________________________________________________________________________________________________________________
Folder Structure:
1. Datasets are saved in a folder called "Dataset". 
2. Then Subfolders indicate the specific dataset, so e.g. "Cut".
3. Each of these has subfolders called [Dataset]_Velocity, [Dataset]_Density, [Dataset]_TCD (Topological charge density)
4. In each of these folders the according data is stored in a .csv file. For each timepoint exists one file, which is named [Dataset]_[10000+timepoint].csv

______________________________________________________________________________________________________________________________________________________________________
Quick descriptions of the main functions:

> velocity_csv()  
Only works on the "Cut" and "Delta_Catenin" datasets. Calculates the velocities in x (named u) and y (named v) for individual cells. Also uses a                         Savinsky-Golay-filter to smoothen trajectories. Saves "x", "y", "u" and "v" in a .csv file. Each csv file corresponds to a single timepoint, which contains all cells. 

> velocity_interpolation()
Interpolates the velocity obtained in velocity_csv() onto a grid. Saves again  "x", "y", "u" and "v" in a .csv file. Each csv file corresponds to a single timepoint, which contains all cells. 

> load_data()
Used to calculate the cell density in each grid point. Also shifts the velocities into the middle of each grid-cell

> topological_quantities()
Calculates the topological charge density rho, and the flux in x and y direction (Flux = rho * velocity)

> experimental_charge_velocity()
Iterates through all time points and calculates the "topological quantities" from the previous function and saves them in a .csv.  
