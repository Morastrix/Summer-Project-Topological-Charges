import os
import Functions


#-------------------------------------------------------------------------------------------
""" Hello, bienvenido, velkommen and Willkommen, first of, you start with both python files together with your data 
    in a single folder. We are now going to create a Datasets folder, which is used as a storage space for all csv
    data files. Please deside how you want to call this dataset                                                         """
Dataset = 'Cut'                             # The Dataset you are currently working on 

script_dir = os.path.dirname(__file__)
general_dir = os.path.join(script_dir, 'Datasets')
dataset_dir = os.path.join(general_dir, Dataset)

if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
""" Now, please move the .csv file storing the cell tracking data into the Datasets folder and in there into the folder
    which has the name you gave this dataset. Thank you :) Please copy the file name into the variable beneath, but without
    the .csv ending (Ideally just your Dataset name)                                                                    """

original_file = 'Cut'

""" Give a name to the dataset you are working on, so you can remember which one it is. Also try to find what the area is 
    in which the cells are moving and enter the limits into xy_extent this can be done using the functions below.
    However, you don't have to use the exact values returned from the function. The limits are later used for fitting the data 
    onto a grid with these limit. I would suggest using round values                                                    """

#Functions.find_limits(Dataset)

xy_extent = [0, 720, 0, 2240]               # xmin, xmax, ymin, ymax of the dataset

#-------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------
""" If Cell-tracking data is given, use this function to extract individual cell velocities 
    (Just a hint, this can take a bit)                                                                                  """

Density_folder = 'Cut_Density'          # In a folder with this name the individual cell velocities will be stored

#Functions.velocity_extraction(Dataset, Density_folder) 
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
""" Well done, now after we hae the velocities for each individual cell, we want to interpolate them onto a grid
    To smoothen the velocities, we use a gaussian filter, which is the last parameter of the next function. 
    Generally, I would advise a large sigma (~10 or higher)                                                             """

Velocity_folder = 'Cut_Velocity_10'      # How to name the Folder, where you store the velocities interpolated onto a grid
Gridspacing = 10                         # Defines the gab between each gridspace   
Sigma = 10                               # Defines the extent of the gaussian filter

#Functions.velocity_interpolation(Dataset, Gridspacing, xy_extent, Density_folder, Velocity_folder, test_sigma=Sigma)

""" If you want a video, displaying these velocities now, use the following function (Step size reduces the
    amount of arrows displayed in this case):                                                                           """

step_size = 1                           # Defines by how much the amount of arrows displayed are reduced (only certain ones are defined)
Velocity_plots_folder = 'Quiver_Velocity'# How to name the folder where the images are stored

#Functions.cell_velocity_plots_iterate(Dataset, Velocity_folder, Velocity_plots_folder, step_size) # Could give issues if file is too large
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
""" You look dazzling today. Next step in line is to calculate the topological charge density, which is done with the
    following functions. Don't worry, you don't have to think to much here. The norm is used so that we can set 
    normalize the values for displaying them in the following function                                                  """

TCD_folder = 'TCD'              # Name of the folder, where the topological charge densities are stored 
norm = 44119522309.78328        # Update this value according to the output below, if you don't want to run the function again

#norm = Functions.charge_density_csv_iterate(Dataset, Velocity_folder, TCD_folder)

""" If you loved the video from before, you for sure are going to enjoy this one. This one shows shows the Topological
    Charge Density. Enticing, isn't it?                                                                                 """ 
    
plot_charges_folder = 'Charges'  # Folder to save the images of the charges                                                                      

#Functions.charge_density_plots_iterate(Dataset, norm, xy_extent, Velocity_folder, TCD_folder, plot_charges_folder)
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
""" Have you taken a break today? If not take five minutes to walk around, give your co-worker a compliment or grab 
    a cup of coffee. Thereafter, we want to find where our defects are located. Use the function beneath to do this.     """

plot_detection_folder = 'Detection'     # If you want images of the densities, with the defect position shown in the images
threshold = 0.4                         # Might need to play around with this, needs to be between 0 and 1 (in units of max value) 

#Functions.defect_position_csv_iterate(Dataset,threshold, norm, Gridspacing, xy_extent, TCD_folder, plot_detection_folder)

""" This one is not quite working yet, but it's a start. This one tries to display the trajectories of the defects      """

#Functions.defect_trajectory_plot(Dataset, f'Defect_{TCD_folder}')
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
""" The following functions are an attempt to calculate the topological current according to the Skogvoll paper          """
#Functions.iterate_theo_charge(Dataset, Velocity_folder, 'Charge_center_6')
#Functions.iterate_charge_plots(Dataset,'Charge_center_6','Defect_Sigma_6','TC_Velocities_Sigma_6', 379, 5, 5, part=2)
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
""" This can be used when dealing with a new dataset, to inspect individual cell trayectories, but is generally
    not an important function                                                                                             """

cell_number = [18612]

#Functions.individual_trajectories(Dataset, cell_number)
#-------------------------------------------------------------------------------------------

print('All done')