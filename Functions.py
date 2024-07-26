import numpy as np
import os
import cv2
import csv
from tqdm import tqdm
import fnmatch
import gc
import psutil

from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy.ndimage import label, center_of_mass, gaussian_filter

import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
import warnings

script_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(script_dir, 'Datasets')

# Path to the 'Results' folder, where all the results (Images) are saved. If this path does not exits, it will be created
def result_dir(Dataset):
        results_path = os.path.join(script_dir, 'Results')
        results_path = os.path.join(results_path, Dataset)
        if not os.path.isdir(results_path):
                os.makedirs(results_path)
        return results_path

def plot_dir(Dataset, save_folder):
        save_path = os.path.join(result_dir(Dataset), save_folder)

        if not os.path.isdir(save_path):
                os.makedirs(save_path)
        return save_path


#-------------------
# General functions
#-------------------


def find_limits(Dataset):
        """
        Used for finding the limits of the dataset, used for defining a frame of the dataset.
        
        Parameters:
        -----------------
        Dataset : str
                Name of the dataset, e.g. 'Cut'
        """
        general_path = os.path.join(dataset_dir, Dataset)
        file_path = os.path.join(general_path, f'{Dataset}.csv') # Path to the cell-tracked .csv file (original data)

        track_ID, x, y, t =  np.loadtxt(file_path, delimiter=",", skiprows=4, usecols = (2,4,5,7), unpack=True)

        print(f'xmin: {np.min(x)}, xmax: {np.max(x)}.')
        print(f'ymin: {np.min(y)}, ymax: {np.max(y)}.')
        
        return 0


def TC_without_density(Dataset, file_velocity, save_path=''):
        """
        Dataset:
                'Cut' or if a new one name of the dataset which is used
        file_velocity : str
                Folder and file to the velocity
        
        """
        # Load velocity data
        velocity_path = os.path.join(r'C:\Users\User\Desktop\Physikstudium\Masterstudium\Summer Project\Datasets', Dataset)
        velocity_path = os.path.join(velocity_path, file_velocity)
        x,y,u,v = np.loadtxt(velocity_path, delimiter=",", skiprows=1, unpack=True)

        # Replace Nan-values in the velocity files with 0.0
        u = np.nan_to_num(u)
        v = np.nan_to_num(v)

        # Multiply the velocities by 10^10 to avoid numerical errors
        u = u*10**10
        v = v*10**10

        # Get the figure shape
        fig_shape_x = np.shape(np.unique(x))[0]
        fig_shape_y = np.count_nonzero(x == x[0])
        fig_shape = (fig_shape_y, fig_shape_x)

        u = u.reshape(fig_shape)
        v = v.reshape(fig_shape)

        dx = np.unique(x)[1]-np.unique(x)[0]
        dy = np.unique(y)[1]-np.unique(y)[0] 

        dU_dx = np.gradient(u, dx, axis=0)
        dU_dy = np.gradient(u, dy, axis=1)
        dV_dx = np.gradient(v, dx, axis=0)
        dV_dy = np.gradient(v, dy, axis=1)

        k = np.multiply(dU_dx, dV_dy)-np.multiply(dV_dx, dU_dy)
        rho = k/np.pi

        # Used for finding the max value later one for normalizing the plots
        max_density = np.max(np.abs(rho))

        if save_path:

                rho = rho.ravel()
                u = u.ravel()
                v = v.ravel()

                data = np.column_stack((x, y, rho, u, v))
                with open(save_path, 'w', newline='') as csvfile:
                        # Create a csv.writer object
                        writer = csv.writer(csvfile)
                        # Write data
                        writer.writerow(['x', 'y', 'Rho', 'u', 'v'])
                        writer.writerows(data)

        return max_density


def measurement_filter(x_cells, y_cells, u_cells=np.array([]), v_cells=np.array([])):
    """
    Filter out measurements with y-values far above the others. Also includes velocity values

    Parameters
    ----------
    x_cells: array
        x-coordinates of the cells
    y_cells: array
        y-coordinates of the cells
    u_cells: array (optional)
        u-coordinates of the cells 
    v_cells: array (optional)
        v-coordinates of the cells
    """

    # Filter out any cells being far from the other cells (probably measurement errors?)
    indices_largest = np.argpartition(y_cells, -20)[-20:] # Select the 20 largest values of y
    largest = y_cells[indices_largest]
    mean = np.mean(largest) # Calculate the mean value of the 20 largest values
    indices_over_thres = np.where(largest > mean+60) 

    if np.any(indices_over_thres[0]):

        min_delete = np.min(largest[indices_over_thres]) # Find the minimum value, above which all values get deleted
        indices_delete = np.where(y_cells >= min_delete) 
        x_cells = np.delete(x_cells, indices_delete)
        y_cells = np.delete(y_cells, indices_delete)
        if u_cells.any():
                u_cells = np.delete(u_cells, indices_delete)
                v_cells = np.delete(v_cells, indices_delete) 

    if u_cells.any():
        return x_cells, y_cells, u_cells, v_cells
    else:
        return x_cells, y_cells


def create_csv(Dataset, time_points, use='', header=[], folder_name='Velocity'):
        """
        Creates a csv file for each time point

        Parameters
        ----------
        Folder: str
                Folder where these csv files are created in 
        use: str
                Either 'Velocity', 'Velocity_Interpolation' or 'Topology'
        header: array
                writes a headline into the file 'x,y,u,v'
        """
        # Sorry for the mess...

        path = os.path.join(dataset_dir, Dataset)
        if use == 'Velocity_Interpolation':
                Folder = os.path.join(path, folder_name)
        elif use == 'Velocity':
                Folder = os.path.join(path, f'{Dataset}_Density')       
        else:
                raise Exception("Select a correct use, either 'Velocity_Interpolation' or 'Velocity'.")
                return -1
        
        for time in range(0, time_points):
                csv_name = f'{Dataset}_{int(10000+time)}.csv'
                csv_file_path = os.path.join(path, Folder)
                csv_file_path = os.path.join(csv_file_path, csv_name)

                with open(csv_file_path, 'w', newline='') as csvfile:
                        fieldnames = ['x', 'y', 'u', 'v']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if header:
                                writer.writeheader()


def velocity_extraction(Dataset, density_folder):
        """
        Uses the given data consisting of track_ID, x, y and t to calculate individual cell velocities and store them in a csv file 
        according to the correct time. (Used for Cut dataset). To lower the noise it uses every 10th timepoint of a particle for 
        calculation of the cell path.

        Parameters
        ----------
        Dataset : str
                'Cut' or 'Delta_Catenin', depending on what you use
        density_folder : str
                Where the interpolated velocities should be stored, folder will then be created
                Density, because it returns the cells with individual velocities, but not on a grid
        """        

        general_path = os.path.join(dataset_dir, Dataset)
        densiy_path = os.path.join(general_path, density_folder)        
        file_path = os.path.join(general_path, f'{Dataset}.csv') # Path to the cell-tracked .csv file (original data)

        if not os.path.isdir(densiy_path):
                os.makedirs(densiy_path)

        track_ID, x, y, t =  np.loadtxt(file_path, delimiter=",", skiprows=4, usecols = (2,4,5,7), unpack=True)
        delta_t = np.round(np.unique(t))[1]
        time_points = np.round(np.unique(t/delta_t)).astype(int)[-1]

        create_csv(Dataset, time_points, use='Velocity', header=['x', 'y', 'u', 'v']) # Creates csv files for each timepoint

        used_cells = 0  # Counts how many cells are used for the velocity extraction
        interval = 10   # If a cell has more datapoints than the interval number, it will be used

        # Iterates through all track_IDs and write x, y, u, v in a csv file according to their timepoint.
        for elem in tqdm(np.unique(track_ID), desc="Progress"): # Selects individual cells
                # Finds indices with the selected track_ID
                indices = np.where(track_ID == elem)
                x_elem = x[indices]
                y_elem = y[indices]
                t_elem = t[indices]

                if np.shape(t_elem)[0] > interval:
                        
                        time_columns = np.shape(t_elem)[0] - interval
                        if time_columns > 10: time_columns = 10

                        used_cells += 1
                        sort_index = np.argsort(t_elem)

                        x_elem = x_elem[sort_index]
                        y_elem = y_elem[sort_index]
                        t_elem = t_elem[sort_index]
                        
                        for time_fragment in range(time_columns):
                                x_elem_int = x_elem[time_fragment::interval]
                                y_elem_int = y_elem[time_fragment::interval]
                                t_elem_int = t_elem[time_fragment::interval]


                                window_length = np.shape(t_elem_int)[0]
                                if window_length >= 3:
                                        if window_length <= 12:
                                                polyorder = window_length-2
                                        elif window_length > 12:
                                                polyorder = 10
                                        x_elem_int = savgol_filter(x_elem_int, window_length=window_length, polyorder=polyorder)
                                        y_elem_int = savgol_filter(y_elem_int, window_length=window_length, polyorder=polyorder)


                                u = np.gradient(x_elem_int, t_elem_int)
                                v = np.gradient(y_elem_int, t_elem_int)

                                file_number = np.round(np.unique(t_elem_int/delta_t)).astype(int)  
                                # Go through all times, find the corresponding csv file given by 'time' and write x,y,u,v data in this file

                                for idx, time in enumerate(file_number):
                                        csv_name = f'{Dataset}_{int(10000+time)}.csv'
                                        csv_file_path = os.path.join(densiy_path, csv_name)
                                        with open(csv_file_path, 'a', newline='') as csvfile:
                                                # Create a csv.writer object
                                                fieldnames = ['x', 'y', 'u', 'v']
                                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                                writer.writerow({'x': x_elem_int[idx], 'y': y_elem_int[idx], 'u': u[idx], 'v' : v[idx]})

        print(f'Cells used in total now: {used_cells}')


def velocity_interpolation(Dataset, gridspace, xy_extent, density_folder, velocity_folder, test_sigma=0.0):
        """
        Uses the calculated velocity vector fields to interpolate in between values and match them to a grid

        Parameters
        ----------
        Dataset : str
                'Cut', depending on what you use
        gridspace: int
                How large is the spacing supposed to be in between each gridpoint (for the Cut dataset 10 was useful)
        xy_extent: array
                array of the form [xmin, xmax, ymin, ymax], value limits of the datapoints
        density_folder: str
                Name of the folder where individual cell velocities are stored (obtained from velocity_extraction)
        velocity_folder: str
                Name of the folder where the grid velocities should be stored
        test_sigma: float
                To smoothen the data a gaussian filter is applied. This gives the sigma to the gaussian filter.
                Ideally a large value is chosen, e.g. 10.
        """  
        # Path to the Folders
        general_path = os.path.join(dataset_dir, Dataset)               # Path to Dataset folder
        density_path = os.path.join(general_path, density_folder)       # Path to Density folder
        velocity_path = os.path.join(general_path, velocity_folder)     # Path to the desired saving space of velocities
        data_file = os.path.join(general_path, f'{Dataset}.csv')        # Cell-tracking data (Original data)

        if not os.path.isdir(velocity_path):                            # If the velocity folder does not exist, it creates one
                os.makedirs(velocity_path)

        # To figure out how many .csv files need to be created
        t = np.loadtxt(data_file, delimiter=",", skiprows=4, usecols = (7), unpack=True)
        delta_t = np.round(np.unique(t))[1]
        time_points = np.round(np.unique(t/delta_t)).astype(int)[-1]

        create_csv(Dataset, time_points, use='Velocity_Interpolation', folder_name = velocity_folder)

        for i in tqdm(range(0,time_points)):
                filename = f'{Dataset}_{10000+int(i)}.csv'
                density_data = os.path.join(density_path, filename)
                velocity_data = os.path.join(velocity_path, filename)

                x_cells, y_cells, u_cells, v_cells = np.loadtxt(density_data, delimiter=",", skiprows=1, unpack=True)

                # Filter out NaN values in the u & v data 
                valid_mask = ~np.isnan(u_cells) & ~np.isnan(v_cells)
                x_cells = x_cells[valid_mask]
                y_cells = y_cells[valid_mask]
                u_cells = u_cells[valid_mask]
                v_cells = v_cells[valid_mask]

                # Filter out measurement values far off the rest
                x_cells, y_cells, u_cells, v_cells = measurement_filter(x_cells, y_cells, u_cells=u_cells, v_cells=v_cells)                

                # Create the grid (yes it looks fcking weird, but thats how the other data looks like) 
                grid_x, grid_y = np.meshgrid(np.linspace(xy_extent[0], xy_extent[1], int((xy_extent[1]-xy_extent[0])/gridspace+1)), np.linspace(xy_extent[3], xy_extent[2], int((xy_extent[3]-xy_extent[2])/gridspace+1)))

                # Interpolation
                coordinates = np.array([x_cells, y_cells]).T
                grid_u = griddata(coordinates, u_cells, (grid_x, grid_y), method='cubic')
                grid_v = griddata(coordinates, v_cells, (grid_x, grid_y), method='cubic')

                grid_u = np.nan_to_num(grid_u)
                grid_v = np.nan_to_num(grid_v)

                # Gaussian Filter, to smoothen the velocities
                grid_u = gaussian_filter(grid_u, sigma=test_sigma)
                grid_v = gaussian_filter(grid_v, sigma=test_sigma)

                # Saving data
                grid_x = grid_x.ravel() 
                grid_y = grid_y.ravel() 
                grid_u = grid_u.ravel() 
                grid_v = grid_v.ravel()    

                data = np.column_stack((grid_x, grid_y, grid_u, grid_v))

                with open(velocity_data, 'w', newline='') as csvfile:
                        # Create a csv.writer object
                        writer = csv.writer(csvfile)
                        # Write header
                        writer.writerow(['x', 'y', 'u', 'v'])
                        # Write data
                        writer.writerows(data)


def individual_trajectories(Dataset, cell_number):
        """
        Plots trajectories from individual cells, also with a savgol filter, to evaluate the smoothness of the path

        Parameters
        ----------
        Dataset : str
            'Cut' or 'Delta_Catenin'
        cell_number : int / array
                Uses the Track_ID to select a specific cell or an array of cells
        
        """
        file_path = os.path.join(dataset_dir,  Dataset)
        file_path = os.path.join(file_path,  'Chunk_Data.csv')

        track_ID, x, y, t =  np.loadtxt(file_path, delimiter=",", skiprows=4, usecols = (2,4,5,7), unpack=True)

        for elem in cell_number: # Selects individual cells
                # Finds indices with the selected track_ID
                indices = np.where(track_ID == elem)
                x_elem = x[indices]
                y_elem = y[indices]
                t_elem = t[indices]

                sort_index = np.argsort(t_elem)

                x_elem = x_elem[sort_index]
                y_elem = y_elem[sort_index]
                t_elem = t_elem[sort_index]


                plt.scatter(x_elem, y_elem, c='orange', marker='x', lw=4, zorder=2, label='Original Cell Trajectory')
                plt.plot(x_elem, y_elem, c=frankfurt, zorder=0, alpha=0.5, lw=3)

                window_length = np.shape(t_elem)[0]
                if window_length >= 3:
                        if window_length <= 12:
                                polyorder = window_length-2
                        elif window_length > 12:
                                polyorder = 10
                        x_elem = savgol_filter(x_elem, window_length=window_length, polyorder=polyorder)
                        y_elem = savgol_filter(y_elem, window_length=window_length, polyorder=polyorder)
                        plt.scatter(x_elem, y_elem, c=lightgreen, marker='x', lw=4, zorder=3, label='Savitzky-Golay Filter')
                        plt.plot(x_elem, y_elem, c=hamburg, zorder=1, alpha=0.5, lw=3)
        
        
        #plt.xlim(0, 720)
        #plt.ylim(0, 2245)
        plt.legend()
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()

        return 0

 
def charge_density_csv_iterate(Dataset, velocity_folder, TCD_folder):
        """
        Create csv data of the charge density to be later used for plotting, isn't that clear??

        Parameters
        ----------
        Dataset: str
                'Cut' or 'Delta_Catenin'  
        velocity_folder: str
                Where the interpolated velocities are stored
        TCD_folder: str
                Where you want the csv data to be stored   
        """        
        general_path = os.path.join(dataset_dir, Dataset)
        TCD_path = os.path.join(general_path, TCD_folder)
        if not os.path.isdir(TCD_path):
                os.makedirs(TCD_path)

        time_points = len(fnmatch.filter(os.listdir(os.path.join(general_path, velocity_folder)), '*.csv'))
        norm_value = 0.0

        # Calculates topological charge density and writes it into files
        for i in tqdm(range(0, time_points), desc="Calculating the Topological Densities and saving them into .csv files"):
                file = f'{Dataset}_{10000+i}.csv'
                save_path = os.path.join(TCD_path, file)
                file_velocity = os.path.join(velocity_folder, file)

                max_value = TC_without_density(Dataset, file_velocity=file_velocity, save_path=save_path)
                if max_value > norm_value:
                        norm_value = max_value
        print('Calculation of topological quantities done')
        print(f'Normalizing plots with the value: {norm_value}')

        return norm_value

#--------------
# Plotting
#--------------

def single_density_plot(Dataset, quantity, xy_extent, title, plot_folder=''):
        """
        Plot displaying a matrix with imshow

        Parameters
        ----------
        Dataset: str
                Name of the dataset, e.g. 'Cut'
        quantity : array
                Data to be displayed, should be a numpy array matrix
        title : str
                Name of the plot
        xedges: array
                extent of the data in x
        yedges: array
                extent of the data in y
        """
        
        fig, ax = plt.subplots(figsize=(7,7))
        
        cmap = plt.get_cmap('seismic')
        norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)    # Center the colors such that white = 0.0, create symmetric scale around it
        pos = ax.imshow(quantity, interpolation=None, extent=xy_extent, norm=norm, cmap=cmap)
        plt.colorbar(pos)
        ax.set_title(title[:-6])

        if plot_folder:
                path = os.path.join(plot_dir(Dataset, plot_folder), f'{Dataset}_{title}.png')
                plt.savefig(path)
                plt.clf()
                plt.close(fig)


def single_density_tracking(Dataset, quantity, xy_extent, threshold, step_size, title, plot_folder=''):
        """
        Displays the topological charge density and identifies the charges, returns the coordinates of thesee charges.
        Charges are identified if the density exceeds a set threshold

        Parameters
        ----------
        Dataset : str
                'Cut'
        quantity : array
                Data to be displayed, should be a numpy array matrix
        threshold : float
                Value between 0 and 1, threshold defines part of the maximum value needed for detecting a topological charge
        step_size: int
                Value used for reducing the dimension earlier. Will be used to put the defects to the original position
        title : str
                Name of the plot
        plot_folder : str
                If a name is given it will create a folder and save the images in there without dispalying them, else it will show them
                without saving them (for testing)
        """
        
        data = np.abs(quantity)

        #threshold_value = threshold*np.max(data)       # Relative value
        threshold_value = threshold                     # Absolute value

        binary_image = data > threshold_value

        # Label connected components
        labeled_image, num_features = label(binary_image)

        # Find the center of mass of each particle
        centroids = center_of_mass(binary_image, labeled_image, range(1, num_features + 1))

        x_defect = [c[0] for c in centroids]
        y_defect = [c[1] for c in centroids]

        x_defect_pos = np.array(x_defect)
        y_defect_pos = np.array(y_defect)
        orientation = np.zeros_like(x_defect_pos)

        # To figure out, whether a defect is +1 or -1
        #for i in range(len(x_defect)):
        #        orientation[i] = quantity[int(x_defect[i]), int(y_defect[i])]
        #orientation = orientation / np.abs(orientation)

        # Bring them back to the actual positions of the original scale
        x_defect_pos = 2250 - (x_defect_pos) * step_size 
        y_defect_pos = (y_defect_pos - 1) * step_size

        if plot_folder:
                plt.figure(figsize=(8, 8))
                cmap = plt.get_cmap('seismic')

                norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)    # Center the colors such that white = 0.0
                pos = plt.imshow(quantity, interpolation=None ,norm=norm, cmap=cmap, extent=xy_extent)
                plt.scatter(y_defect_pos, x_defect_pos, color='lime', s=30, marker='h')

                plt.colorbar(pos)
                plt.title(title)

                path = os.path.join(plot_dir(Dataset, plot_folder), f'{Dataset}_{title}.png')
                plt.savefig(path)
                plt.clf()
                plt.close()
                
        return x_defect_pos, y_defect_pos


def quiver_plot(x, y, u, v, Dataset, title, step_size=1, plot_folder=''):
        """
        Creates a quiver plot

        Parameters
        ----------
        x : array
                x-position of arrows
        y : array
                y-position of arrows 
        u: array
                Arrow-length in x-direction
        v: array
                Arrow-length in y-direction
        Dataset : str
                Name of the dataset, e.g. 'Cut'
        title :  str
                Title of the images to be saved (if multiple are plotted indicate it by numbers here)
        step_size: int
                Reduces amount of arrows by the defined stepsize
        plot_folder: str
                If a name is given it will save the images in the given folder, without dispaying them. If no folder is given it will
                not save the image, but instead dispaly it (useful for testing)
        """
        # Get the figure shape
        fig_shape_y = np.shape(np.unique(x))[0]
        fig_shape_x = np.count_nonzero(x == x[0])
        fig_shape = (fig_shape_x, fig_shape_y)

        x = x.reshape(fig_shape).T
        y = y.reshape(fig_shape).T
        u = u.reshape(fig_shape).T
        v = v.reshape(fig_shape).T

        # If the figure shape is not divisible by the step size, the dimensions get adjusted
        overshoot_dim_x =  (fig_shape[1] % step_size)
        overshoot_dim_y =  (fig_shape[0] % step_size)
                
        if overshoot_dim_x:
                x = x[:-overshoot_dim_x,]
                y = y[:-overshoot_dim_x,]
                u = u[:-overshoot_dim_x,]
                v = v[:-overshoot_dim_x,]
        if overshoot_dim_y:
                x = x[:,:-overshoot_dim_y]
                y = y[:,:-overshoot_dim_y]
                u = u[:,:-overshoot_dim_y]
                v = v[:,:-overshoot_dim_y]


        u = np.nan_to_num(u)
        v = np.nan_to_num(v)

        # Reshape the array to split into chunks of size step x step
        reshaped_x = x.reshape(x.shape[0] // step_size, step_size, x.shape[1] // step_size, step_size)
        reshaped_y = y.reshape(y.shape[0] // step_size, step_size, y.shape[1] // step_size, step_size)
        reshaped_u = u.reshape(u.shape[0] // step_size, step_size, u.shape[1] // step_size, step_size)
        reshaped_v = v.reshape(v.shape[0] // step_size, step_size, v.shape[1] // step_size, step_size)

        # Calculate the mean over each chunk
        x = reshaped_x.mean(axis=(1, 3))
        y = reshaped_y.mean(axis=(1, 3))
        u = reshaped_u.mean(axis=(1, 3))
        v = reshaped_v.mean(axis=(1, 3))

        # Normalize Arrows to the same length
        magnitude = np.sqrt(u**2 + v**2)        # Used for normalization
        magnitude[magnitude == 0] = 1e-10       # Avoids division by 0
        u = u / magnitude
        v = v / magnitude

        plt.figure(figsize=(15,45))

        # Dpeending on which step size is used, the scale should be adjusted to make the arrows visible, yet not overexaggerated
        if step_size == 5:
                scale = 60
        elif step_size == 1:
                scale = 200
        elif step_size == 3:
                scale = 105
        elif step_size == 30:
                scale = 105
        else:
                raise ValueError('Step_size not know, add new scale value in quiver_plot')
                        
        plt.quiver(x, y, u, v, np.arctan2(v, u), cmap='hsv', scale=scale, minshaft=2)

        if plot_folder:
                path = os.path.join(plot_dir(Dataset, plot_folder), f'{Dataset}_{title}.png')
                plt.savefig(path)
                plt.clf()
                plt.close()
        else:
                plt.savefig('Test.pdf')
                plt.show()


def charges_quiver(Dataset,title, plot_folder, Velocities_file, Defect_file, step_x, step_y, normalized=True, save=True):
        """
        Creates a quiver plot of the topological charge velocity

        Parameters
        ----------
        Dataset : str
                'Cut'
        title : str
                Ideally just use the iteration number, to get a series of sorted images
        plot_folder: str
                Folder where the images should be stored
        v: array
                Arrow-length in y-direction
        step: int
                Reduces amount of arrows by the defined stepsize
        threshold: float
                Hides arrows under a certain threshold (not really useful)
        normalized: bool
                Arrows have the same length, cannot be combined with threshold
        """

        x, y, u, v = np.loadtxt(Velocities_file, skiprows=1, usecols=(0,1,2,3), unpack=True, delimiter=',')
        x_defect, y_defect = np.loadtxt(Defect_file, skiprows=1, usecols=(0,1), unpack=True, delimiter=',')
        x_defect = 2240/223*x_defect
        y_defect = 720/71 * y_defect

        # Get the figure shape
        fig_shape_y = np.shape(np.unique(x))[0]
        fig_shape_x = np.count_nonzero(x == x[0])
        fig_shape = (fig_shape_x, fig_shape_y)
        fig_shape = fig_shape

        x = x.reshape(fig_shape).T
        y = y.reshape(fig_shape).T
        u = u.reshape(fig_shape).T
        v = v.reshape(fig_shape).T

        # If the figure shape is not divisible by the step size, the dimensions get adjusted
        overshoot_dim_x =  (fig_shape[1] % step_x)
        overshoot_dim_y =  (fig_shape[0] % step_y)
                
        if overshoot_dim_x:
                x = x[:-overshoot_dim_x,]
                y = y[:-overshoot_dim_x,]
                u = u[:-overshoot_dim_x,]
                v = v[:-overshoot_dim_x,]
        if overshoot_dim_y:
                x = x[:,:-overshoot_dim_y]
                y = y[:,:-overshoot_dim_y]
                u = u[:,:-overshoot_dim_y]
                v = v[:,:-overshoot_dim_y]


        u = np.nan_to_num(u)
        v = np.nan_to_num(v)

        # Reshape the array to split into chunks of size step x step
        reshaped_x = x.reshape(x.shape[0] // step_y, step_y, x.shape[1] // step_x, step_x)
        reshaped_y = y.reshape(y.shape[0] // step_y, step_y, y.shape[1] // step_x, step_x)
        reshaped_u = u.reshape(u.shape[0] // step_y, step_y, u.shape[1] // step_x, step_x)
        reshaped_v = v.reshape(v.shape[0] // step_y, step_y, v.shape[1] // step_x, step_x)

        # Calculate the mean over each chunk
        x = reshaped_x.mean(axis=(1, 3))
        y = reshaped_y.mean(axis=(1, 3))
        u = reshaped_u.mean(axis=(1, 3))
        v = reshaped_v.mean(axis=(1, 3))


        magnitude = np.sqrt(u**2 + v**2)        # Used for normalization
        magnitude[magnitude == 0] = 1e-10       # Avoids division by 0

        plt.figure(figsize=(15,5))              # Use with reduced arrows
        #plt.figure(figsize=(45,15))            # Use when displaying all arrows

        if normalized == True:
                u = u / magnitude
                v = v / magnitude

                if step_x == 5:
                        scale = 60
                elif step_x == 1:
                        scale = 200
                elif step_x == 3:
                        scale = 105
                else:
                        raise ValueError('Step_size not know, add new scale value in quiver_plot')
                
                plt.scatter(x_defect, y_defect, marker="D", color='red')        
                plt.quiver(y, x, v, u, color='grey', scale=scale, minshaft=2)
        else:
                plt.quiver(x, y, u, v)

        if save== True: 
                path = os.path.join(plot_dir(Dataset, plot_folder), f'{Dataset}_{title}.png')
                plt.savefig(path)
                plt.close()
        else:
                plt.savefig('Test.pdf')
                plt.show()


def defect_trajectory_plot(Dataset, defect_folder):
        """
        Function for plotting the defect trajectories. Does noot fully work. Did not figure out how to identify which point belongs
        to which defect. Also restricted to max 4 defects. 
        
        Parameters:
        -------------
        Dataset : str
                'Cut' or a newer one
        defect_folder : str
                Folder, where the positions of the defects are saved in csv files
        """
        general_path = os.path.join(dataset_dir, Dataset)
        folder_path = os.path.join(general_path, defect_folder)

        time_points = len(fnmatch.filter(os.listdir(folder_path), '*.csv'))

        x_collected_1 = np.array([])
        y_collected_1 = np.array([])

        x_collected_2 = np.array([])
        y_collected_2 = np.array([])

        x_collected_3 = np.array([])
        y_collected_3 = np.array([])

        x_collected_4 = np.array([])
        y_collected_4 = np.array([])

        for i in range(time_points):
                defect_file = os.path.join(folder_path, f'{Dataset}_{10000+i}.csv')
                with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)  # Suppress the specific warning
                        try:    
                                x, y =  np.loadtxt(defect_file, delimiter=",", skiprows=1, unpack=True, usecols=(0,1))
                        except (OSError, ValueError):
                                x = np.array([])
                                y = np.array([])

                # There is for sure a better solution for this, feel free to come up with one
                # This exercise is left to solve for the reader... 💓

                if x.size == 0:
                        continue  
                elif x.ndim == 0:
                        x_collected_1 = np.append(x_collected_1, x)
                        y_collected_1 = np.append(y_collected_1, y)
                elif x.ndim == 1 and x.shape[0] == 1:
                        x_collected_1 = np.append(x_collected_1, x)
                        y_collected_1 = np.append(y_collected_1, y)
                else:
                        if np.shape(x)[0] > 4:
                                print('Alarm! Alarm!')
                        if np.shape(x)[0] > 3:
                                x_collected_4 = np.append(x_collected_4, x[-4])
                                y_collected_4 = np.append(y_collected_4, y[-4])
                        if np.shape(x)[0] > 2:
                                x_collected_3 = np.append(x_collected_3, x[-3])
                                y_collected_3 = np.append(y_collected_3, y[-3])
                        if np.shape(x)[0] > 1:
                                x_collected_2 = np.append(x_collected_2, x[-2])
                                y_collected_2 = np.append(y_collected_2, y[-2])
                                x_collected_1 = np.append(x_collected_1, x[-1])
                                y_collected_1 = np.append(y_collected_1, y[-1])

        
        # x and y are twisted to keep it in line with the other images, since imshow turns around x and y
        plt.scatter(y_collected_1, x_collected_1, color='orange', linewidth=3)
        plt.scatter(y_collected_2, x_collected_2, color='orchid', linewidth=3)
        plt.scatter(y_collected_3, x_collected_3, color='lightblue', linewidth=3)
        plt.scatter(y_collected_4, x_collected_4, color='lime', linewidth=3)

        #plt.xlim([0, 2240])
        #plt.ylim([0,720])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


def create_video_from_images(image_folder, output_video, frame_rate):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure the images are in the correct order

    # Read the first image to get the frame size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4 output
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the VideoWriter object
    video.release()


#-----------------
# Multiple Plots:
#-----------------

def cell_velocity_plots_iterate(Dataset, velocity_folder, save_folder, step_size):
        """
        Creates Images and a Video of the velocity plots

        Parameters
        ----------
        Dataset: str
                'Cut' for example
        velocity_folder : str
                Name of the folder which stores the cell velocities interpolated onto a grid
        save_folder : str 
                Name of the folder in which the images should be safed
        step_size : int
                chunks together arrows to display less arrows     
        """

        # Directory Management 
        general_velocity_path = os.path.join(dataset_dir, Dataset)
        density_path = os.path.join(general_velocity_path, velocity_folder)
        time_points = len(fnmatch.filter(os.listdir(density_path), '*.csv'))
        matplotlib.use('Agg') # use this if plotting multiple images, without displaying them

        # Create Plots
        for i in tqdm(range(0, time_points), desc='Creating Cell-Velocity-Quiver-Plots, while you burn your lips sipping coffe that is way to hot'):
                velocity_path = os.path.join(density_path, f'Cut_{10000+i}.csv')
                x, y, u, v = np.loadtxt(velocity_path, delimiter=",", skiprows=1, unpack=True)

                quiver_plot(x, y, u, v,  'Cut', f'{10000+i}', step_size=step_size, plot_folder=save_folder)

                # To avoid memory issues
                if i % 50 == 0: 
                        gc.collect()

        # Create Movie
        print('Be prepared for Robin Hood in a quiver video, shooting arrows...')
        output_video = f'{Dataset}_{save_folder}.mp4'
        frame_rate = 10  # Frames per second
        save_path = plot_dir(Dataset, save_folder)
        create_video_from_images(save_path, output_video, frame_rate)


def charge_density_plots_iterate(Dataset, norm_value, xy_extent, velocity_folder, TCD_folder, plot_folder):
        """
        Stupid function to calculate topological charge densities, write these into a file, plot them and make a video

        Parameters
        ----------
        Dataset: str
                'Cut' or 'Delta_Catenin'  
        velocity_folder: str
                Where the interpolated velocities are stored
        plot_folder: str
                Where you want the images to be stored, folder does not have to exist previously    
        """

        general_path = os.path.join(dataset_dir, Dataset)
        save_TCD_path = os.path.join(general_path, TCD_folder)
        defect_save_folder  = os.path.join(general_path, f'Defect_{TCD_folder}')

        if not os.path.isdir(defect_save_folder):
                os.makedirs(defect_save_folder)

        time_points = len(fnmatch.filter(os.listdir(os.path.join(general_path, velocity_folder)), '*.csv'))

        process = psutil.Process()
        matplotlib.use('Agg') # use this if plotting multiple images, without displaying them

        # Plots images of the Topological charge density and saves them into a folder
        for i in tqdm(range(0, time_points), desc='Generating Topological charge density images'):
                file = f'{Dataset}_{10000+i}.csv'
                TCD_file = os.path.join(save_TCD_path, file)

                rho = np.loadtxt(TCD_file, delimiter=",", skiprows=1, unpack=True, usecols=(2))
                rho = rho.reshape(225, 73)      # Gridspacing = 10
                rho = rho/norm_value            # Normalize the rho value of all images
                
                single_density_plot('Cut', rho, xy_extent, f'Cut_TCD_{10000+i}', plot_folder=plot_folder)

                if i % 50 == 0: 
                        gc.collect()
                        #print(f"Iteration {i}: Memory usage: {process.memory_info().rss / 1024 ** 2} MB")


        # Uses the images from the folder to create a video
        print('Creating Video, Netflix is gonna be jealous')
        output_video = f'{Dataset}_{plot_folder}.mp4'   
        frame_rate = 10  # Frames per second
        image_files = plot_dir(Dataset, plot_folder)
        create_video_from_images(image_files, output_video, frame_rate)

        return 0


def defect_position_csv_iterate(Dataset,threshold, norm_value, step_size, xy_extent, TCD_folder, plot_folder=''):
        """
        Stupid function to calculate topological charge densities, write these into a file, plot them and make a video

        Parameters
        ----------
        Dataset: str
                'Cut' or 'Delta_Catenin'  
        norm_value: int
                Value used for norming the charge density plot
        step_size : int
                step size used in the grid interpolation, to recreate the actual dimensions of the plot
        xy_extent : array
                array of the form [xmin, xmax, ymin, ymax] contains the limits of the plot
        TCD_folder: str
                Name of the folder storing the csv data of the topological charge density
        plot_folder: str
                Where you want the images to be stored, folder does not have to exist previously.
                If none is given data will not be plotted
        """

        general_path = os.path.join(dataset_dir, Dataset)
        TCD_path = os.path.join(general_path, TCD_folder)
        defect_save_folder  = os.path.join(general_path, f'Defect_{TCD_folder}')

        if not os.path.isdir(defect_save_folder):
                os.makedirs(defect_save_folder)

        time_points = len(fnmatch.filter(os.listdir(TCD_path), '*.csv'))

        if plot_folder:
                matplotlib.use('Agg') # use this if plotting multiple images, without displaying them
    
        # Plots images of the Topological charge density and saves them into a folder
        for i in tqdm(range(0, time_points), desc='Generating Topological charge density images'):
                file = f'{Dataset}_{10000+i}.csv'
                TCD_file = os.path.join(TCD_path, file)

                rho = np.loadtxt(TCD_file, delimiter=",", skiprows=1, unpack=True, usecols=(2))
                rho = rho.reshape(225, 73)      # Gridspacing = 10
                rho = rho/norm_value            # Normalize the rho value of all images

                x_defect, y_defect = single_density_tracking('Cut', rho, xy_extent, threshold, step_size, f'TCD_{10000+i}', plot_folder=plot_folder)

                if i % 50 == 0: 
                        gc.collect()
                
                #Save defect position
                defect_file = os.path.join(defect_save_folder, f"Cut_{10000+i}.csv")
                data = np.column_stack((x_defect, y_defect))
                with open(defect_file, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['x','y'])
                        writer.writerows(data)
                
        if plot_folder:
                # Uses the images from the folder to create a video
                print('Creating Video, Netflix is gonna be jealous')
                output_video = f'{Dataset}_{plot_folder}.mp4'   
                frame_rate = 10  # Frames per second
                image_files = plot_dir(Dataset, plot_folder)
                create_video_from_images(image_files, output_video, frame_rate)

        return 0


# --------------------------------------
# Functions, that do not really work 
# --------------------------------------

def topological_current(u, v, u_prev, v_prev, u_next, v_next, dx, dy, shape):
        """
        Theoretically calculates the Charge velocity via
                Vx = -2⋅[(𝜕t Ψ𝑥)⋅(𝜕𝑦 Ψ𝑦) - (𝜕t Ψ𝑦)⋅(𝜕𝑦 Ψ𝑥)]/k
                Vy = -2⋅[(𝜕t Ψ𝑦)⋅(𝜕𝑥 Ψ𝑥) - (𝜕t Ψ𝑥)⋅(𝜕𝑥 Ψ𝑦)]/k
        where 
                k = (𝜕𝑥 Ψ𝑥)⋅(𝜕𝑦 Ψ𝑦) - (𝜕𝑥 Ψ𝑦)⋅(𝜕𝑦 Ψ𝑥)

        Parameters
        ----------
        file_prev: str
                Previous image to be processed
        file: str
                Actual iage to be processed
        file_next: str
                Following image to be processed 
        """

        time_interval = 1       # Welcome to physics 101 lol

        u = np.nan_to_num(u).reshape(shape)
        v = np.nan_to_num(v).reshape(shape)
        u_prev = np.nan_to_num(u_prev).reshape(shape)
        v_prev = np.nan_to_num(v_prev).reshape(shape)
        u_next = np.nan_to_num(u_next).reshape(shape)
        v_next = np.nan_to_num(v_next).reshape(shape)

        du_dx = np.gradient(u, dx, axis=1)
        du_dy = np.gradient(u, dy, axis=0)
        dv_dx = np.gradient(v, dx, axis=1)
        dv_dy = np.gradient(v, dy, axis=0)

        # Calculate the time derivative of the flux using the symmetric difference quotient
        du_dt = (u_next-u_prev)/(2*time_interval)          
        dv_dt = (v_next-v_prev)/(2*time_interval)

        # TODO: Issue with the division by k, since k is 0 at many positions
        #k = np.multiply(du_dx, dv_dy)-np.multiply(dv_dx, du_dy) yep thats bullshit don't use that
        k = 1 # Just set k to 1 lol, what can go wrong, right? 
        current_x = -1/np.pi*(du_dt * dv_dy - dv_dt * du_dy) / k 
        current_y = 1/np.pi*(du_dt * dv_dx - dv_dt * du_dx) / k 

        return current_x, current_y


def iterate_theo_charge(Dataset, Velocity_thing, pre_file):
        """
        Calculates the theoretical topological charge velocities and saves them in a file

        """
        dataset_folder = os.path.join(dataset_dir, Dataset)
        velocity_folder = os.path.join(dataset_folder, Velocity_thing)
        save_pre_file = os.path.join(dataset_folder, pre_file)

        if not os.path.isdir(save_pre_file):
                os.makedirs(save_pre_file)

        create_csv(Dataset, 379, use='Velocity_Interpolation', folder_name = 'Vector_Charges')

        # Load in first file
        first_file = os.path.join(velocity_folder, f'{Dataset}_10000.csv')
        u_prev, v_prev = np.loadtxt(first_file, unpack=True, usecols=(2,3), delimiter=',', skiprows=1)

        second_file = os.path.join(velocity_folder, f'{Dataset}_10001.csv')
        u, v, x, y = np.loadtxt(second_file, unpack=True, usecols=(2,3,0,1), delimiter=',', skiprows=1)

        # Spatial derivatives of the actual image
        dx = np.unique(x)[1]-np.unique(x)[0]
        dy = np.unique(y)[1]-np.unique(y)[0] 

        # Only calculates from the second image to the second to last
        for i in tqdm(range(1,378), desc='Calculating Velocities of TC, so shut up and let me work'):

                velocity_file = os.path.join(velocity_folder, f'Cut_{10001+i}.csv')
                u_next, v_next = np.loadtxt(velocity_file, unpack=True, usecols=(2,3), delimiter=',', skiprows=1)

                charge_x, charge_y = topological_current(u, v, u_prev, v_prev, u_next, v_next, dx, dy, shape=(225,73))
                charge_x = charge_x.ravel()
                charge_y = charge_y.ravel()

                u_prev, v_prev = u, v   # To not load in each dataset 3 times, just repalce the old u and v's with newer ones.         
                u, v = u_next, v_next

                # Saving data
                data = np.column_stack((x, y, charge_x, charge_y))
                save_name = f'{Dataset}_{10001+i}.csv'
                save_file = os.path.join(save_pre_file, save_name)

                with open(save_file, 'w', newline='') as csvfile:
                        # Create a csv.writer object
                        writer = csv.writer(csvfile)
                        # Write header
                        writer.writerow(['x', 'y', 'charge_x', 'charge_y'])
                        # Write data
                        writer.writerows(data)


def iterate_charge_plots(Dataset,vector_charges_dir,defect_dir, plot_path, t_lim, step_x, step_y, part):
        """
        Iterates through all timepoints to create quiver plots of the topological charge velocity and scattred into these the charges
        Uses the function chargers_quiver for creating the plots.

        Parameters
        ----------
        Dataset: str
                'Cut' or 'Delta_Catenin'
        data_folder : str
                Name of the folder to the interpolated velocity data
        save_folder : str 
                Name of the folder in which the images should be safed
        t_lim : int
                amount of timepoints
        step_x : int
                chunks together arrows in x direction, to display less arrows 
        step_y : int       
                chunks together arrows in y direction, to display less arrows     
        """

        # Directory Management
        general_path = os.path.join(dataset_dir, Dataset)
        vector_charges_folder = os.path.join(general_path, vector_charges_dir)
        defect_folder = os.path.join(general_path, defect_dir)
        plot_folder = os.path.join(result_dir(Dataset), plot_path)

        if part == 1:
                start = 2
                end = 250
        if part == 2:
                start = 250
                end = t_lim-2

        for i in tqdm(range(start, end), desc='Calculating images 1/2, take a coffe break'):
                velocity_file = os.path.join(vector_charges_folder, f'{Dataset}_{10000+i}.csv')
                defect_file = os.path.join(defect_folder, f'{Dataset}_{10000+i}.csv')
                charges_quiver('Cut', f'{10000+i}',plot_folder=plot_folder,Velocities_file=velocity_file,Defect_file=defect_file, step_x=step_x, step_y=step_y, save=True)

        if part == 2:
                # Create Movie
                output_video = f'{Dataset}_non_smoothed_6.mp4'
                frame_rate = 10  # Frames per second
                save_path = plot_dir(Dataset, plot_folder)
                create_video_from_images(save_path, output_video, frame_rate)