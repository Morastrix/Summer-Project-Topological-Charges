import numpy as np
import pandas as pd
import os
import cv2
import csv
from tqdm import tqdm
import fnmatch

from scipy.interpolate import griddata
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter
from scipy.misc import derivative
from skimage import filters, feature, morphology
from scipy.ndimage import binary_fill_holes

from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors

from scipy.ndimage import label, center_of_mass, gaussian_filter
from skimage import measure
from skimage.filters import threshold_otsu
script_dir = os.path.dirname(__file__)
dataset_dir = r'C:\Users\User\Desktop\Physikstudium\Masterstudium\Summer Project\Datasets'

# Path to the script and to the folder for the results
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

def load_data(file, Dataset, uniform_density=True, measurement_filtering=False, filtering=None):
        """
        Loads in the data and pre-processes it 

        Parameters
        ----------
        file : str
            Select the file for data extraction
        fig_shape: (a,b)
            Dimensions of the grid
        measurement_filtering: bool
            If True, cells far off the other cells will be filtered out (Used for Cut dataset)
        filtering : str, optional
            To spread cells over multiple pixels, either 'gaussian' or 'uniform', default is no filter
        """

        # Define File Paths
        if Dataset == 'C07' or 'BasicRadial':
                file_velocity = os.path.join(Dataset, f'{Dataset}_Velocity')
                file_velocity = os.path.join(file_velocity, file)

        elif Dataset == 'Cut' or Dataset == 'C05' or Dataset == 'C06' or Dataset == 'Delta_Catenin':
                file_velocity = os.path.join(Dataset, f'{Dataset}_Velocity')
                file_velocity = os.path.join(file_velocity, file)
                file_density = os.path.join(Dataset, f'{Dataset}_Density')
                file_density = os.path.join(file_velocity, file)

        else:
                raise Exception("Select a Dataset, either 'C05', 'C06', 'C07', 'Cut', 'Delta_Catenin', or 'BasicRadial'.")
                return -1

        # Load velocity data
        velocity_path = os.path.join(r'C:\Users\User\Desktop\Physikstudium\Masterstudium\Summer Project\Datasets', file_velocity)
        x,y,u,v = np.loadtxt(velocity_path, delimiter=",", skiprows=1, unpack=True)

        # Replace Nan-values in the velocity files with 0.0
        u = np.nan_to_num(u)
        v = np.nan_to_num(v)

        # Get the figure shape
        fig_shape_x = np.shape(np.unique(x))[0]
        fig_shape_y = np.count_nonzero(x == x[0])
        fig_shape = (fig_shape_x, fig_shape_y)

        u = u.reshape(fig_shape)
        v = v.reshape(fig_shape)

        # Load density data
        if not uniform_density:
                density_path = os.path.join(r'C:\Users\User\Desktop\Physikstudium\Masterstudium\Summer Project\Datasets', file_density)
                cell_x, cell_y = np.loadtxt(density_path, delimiter=",", skiprows=1, unpack=True, usecols=(0,1))

                if measurement_filtering:
                        cell_x, cell_y = measurement_filter(cell_x, cell_y)
                # Calculate bins and amount of cells per bin
                Binned_cells, xedges, yedges = np.histogram2d(cell_x, cell_y, bins=(np.unique(x), np.unique(y)))
                Binned_cells = Binned_cells

                # Spread out cells more by using a gaussian filter, is a uniform distribution maybe better? 
                # Also does not (really) take into account the distance to a cell'
                spread = 2.0
                if filtering == 'gaussian':
                        Binned_cells = gaussian_filter(Binned_cells, sigma=spread)
                if filtering == 'uniform':
                        Binned_cells = uniform_filter(Binned_cells, size=spread)

        elif uniform_density:
                Binned_cells = np.ones((fig_shape[0]-1, fig_shape[1]-1))
                xedges = [np.min(x), np.max(x)]
                yedges = [np.min(y), np.min(y)]

        u_middle = np.zeros((fig_shape[0]-1, fig_shape[1]-1))
        v_middle = np.zeros((fig_shape[0]-1, fig_shape[1]-1))
        x_middle = x.reshape(fig_shape)
        x_middle = x_middle[0:fig_shape[0]-1, 0:fig_shape[1]-1] + 0.5*(np.unique(x)[1]-np.unique(x)[0])
        y_middle = y.reshape(fig_shape)
        y_middle = y_middle[0:fig_shape[0]-1, 0:fig_shape[1]-1] + 0.5*(np.unique(y)[1]-np.unique(y)[0])
        # Reshape the velocity field of u and v, since they are defined at the field edges, thus the velocity in the middle of each bin needs to be found
        for i in range(np.shape(u)[0]-1):
                for j in range(np.shape(u)[1]-1):
                        u_middle[i][j] = (u[i,j]+u[i+1,j]+u[i,j+1]+u[i+1,j+1])/4
                        v_middle[i][j] = (v[i,j]+v[i+1,j]+v[i,j+1]+v[i+1,j+1])/4 

        return Binned_cells, u_middle, v_middle, x_middle, y_middle, xedges, yedges


def TC_with_density(file, Dataset, measurement_filtering=False, filtering='uniform', save_path=''):
        """
        Calculates the absolute value of the flux, the topological density and the velocity of the topological charges

        Parameters
        ----------
        file: str
                The image to be processed
        measurement_filtering: str, optional
                Filter used to spread out cells over more than 1 pixel, either None, 'uniform', or 'gaussian' 
        save_path: 
                Path to the txt file to safe data to
        """

        Binned_cells, u_middle, v_middle, x, y, xedges, yedges = load_data(file, Dataset, measurement_filtering=measurement_filtering, filtering=filtering)

        # Define velocity = √(u²+v²)
        #velocity = np.sqrt(np.multiply(u_middle, u_middle)+np.multiply(v_middle, v_middle))

        # Flux = Velocity ⋅ Density
        #flux = np.multiply(Binned_cells, velocity)

        flux_u = np.multiply(Binned_cells, u_middle)
        flux_v = np.multiply(Binned_cells, v_middle)

        dx = np.unique(x)[1]-np.unique(x)[0]
        dy = np.unique(y)[1]-np.unique(y)[0] 

        dx_flux_u = np.gradient(flux_u, dx, axis=1)
        dy_flux_u = np.gradient(flux_u, dy, axis=0)
        dx_flux_v = np.gradient(flux_v, dx, axis=1)
        dy_flux_v = np.gradient(flux_v, dy, axis=0)


        # Charge density according to Skogvoll2024
        # ρ = (𝜕𝑥 Ψ𝑥)⋅(𝜕𝑦 Ψ𝑦) - (𝜕𝑥 Ψ𝑦)⋅(𝜕𝑦 Ψ𝑥) / π, or
        # ρ = k / π with k = (𝜕𝑥 Ψ𝑥)⋅(𝜕𝑦 Ψ𝑦) - (𝜕𝑥 Ψ𝑦)⋅(𝜕𝑦 Ψ𝑥)
        norm = np.multiply(flux_u, flux_u)+np.multiply(flux_v, flux_v) 
        # TODO: Does not work, since the value outside the dish is 0, thus division by 0
        k = np.multiply(dx_flux_u, dy_flux_v)-np.multiply(dx_flux_v, dy_flux_u)
        rho = k/np.pi

        if save_path:
                x = x.ravel()
                y = y.ravel()
                rho = rho.ravel()
                Binned_cells = Binned_cells.ravel()
                flux_u = flux_u.ravel()
                flux_v = flux_v.ravel()

                data = np.column_stack((x, y, rho, Binned_cells, flux_u, flux_v))
                with open(save_path, 'a', newline='') as csvfile:
                        # Create a csv.writer object
                        writer = csv.writer(csvfile)
                        # Write data
                        writer.writerow(['x', 'y', 'Rho', 'Binned_cells', 'Flux_u', 'Flux_v'])
                        writer.writerows(data)

        return rho, Binned_cells, flux_u, flux_v


def TC_without_density(Dataset, file_velocity, save_path=''):
        """
        
        
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

        # Get the figure shape
        fig_shape_x = np.shape(np.unique(x))[0]
        fig_shape_y = np.count_nonzero(x == x[0])
        fig_shape = (fig_shape_x, fig_shape_y)
        fig_shape = (fig_shape[1],fig_shape[0]) # I Fxcked up the dimesions somewhere

        u = u.reshape(fig_shape)
        v = v.reshape(fig_shape)

        dx = np.unique(x)[1]-np.unique(x)[0]
        dy = np.unique(y)[1]-np.unique(y)[0] 

        #u_dx = np.gradient(u, dx, axis=1)
        #u_dy = np.gradient(u, dy, axis=0)
        #v_dx = np.gradient(v, dx, axis=1)
        #v_dy = np.gradient(v, dy, axis=0)
        dU_dx, dU_dy, dV_dx, dV_dy = adaptive_gradient(u, v, dx, dy)
        #print(np.shape(dU_dx))

        k = np.multiply(dU_dx, dV_dy)-np.multiply(dV_dx, dU_dy)
        rho = k/np.pi
        #print(np.shape(rho))
        x = x.reshape(fig_shape)
        y = y.reshape(fig_shape)
        #print(np.shape(x))
        rho = rho[1:-1, 1:-1] # Only if the NumPy Gradient is used
        x = x[1:-1, 1:-1]
        y = y[1:-1, 1:-1]
        u = u[1:-1, 1:-1]
        v = v[1:-1, 1:-1]
        if save_path:
                x = x.ravel()
                y = y.ravel()
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
        return 0


def theoretical_charge_velocity(u, v, u_prev, v_prev, u_next, v_next, dx, dy, shape):
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

        time_interval = 1

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

        # TODO: Derivative is missing at the edges, so first frame and last frame
        # TODO: Not super accurate derivative and quite slow for multiple images
        # Calculate the time derivative of the flux using the symmetric difference quotient
        du_dt = (u_next-u_prev)/(2*time_interval)          
        dv_dt = (v_next-v_prev)/(2*time_interval)

        # TODO: Issue with the division by k, since k is 0 at many positions
        #k = np.multiply(du_dx, dv_dy)-np.multiply(dv_dx, du_dy)
        k = 1 # Just set k to 1 lol, what can go wrong, right? 
        charge_vx = -2*(du_dt * dv_dy - dv_dt * du_dy) / k 
        charge_vy = -2*(dv_dt * du_dx - du_dt * dv_dx) / k 

        return charge_vx, charge_vy


def Nine_point_stencil(psi, dx, dy):
        """
        Self-defined stencil as a gradient method
        
        Parameters:
        -------------
        psi : array
                The corresponding field in x and y-direction
        dx : float
                Stepsize in x-direction
        dy : float
                Stepsize in y-direction
        """

        grad_x = np.zeros((psi.shape[0]-1, psi.shape[1]-1))
        grad_y = np.zeros((psi.shape[0]-1, psi.shape[1]-1))

        # Gradient from the paper
        #grad_x = 1/(8*dx)*(psi[2:,2:] + 2 * psi[1:-1,2:] - psi[:-2,2:] + psi[2:,:-2] - 2 * psi[1:-1,:-2] - psi[:-2,:-2])
        #grad_y = 1/(8*dy)*(psi[2:,2:] + 2 * psi[2:,1:-1] - psi[2:,:-2] + psi[:-2,2:] - 2 * psi[:-2,1:-1] - psi[:-2,:-2])

        # Coorected gradient
        grad_x = 1/(8*dy)*(psi[2:,2:] + 2 * psi[2:,1:-1] + psi[2:,:-2] - psi[:-2,2:] - 2 * psi[:-2,1:-1] - psi[:-2,:-2]) # (i-g) + 2*(f-d) + (c-a)
        grad_y = 1/(8*dx)*(psi[2:,2:] + 2 * psi[1:-1,2:] + psi[:-2,2:] - psi[2:,:-2] - 2 * psi[1:-1,:-2] - psi[:-2,:-2]) # (i-c) + 2*(h-b) + (g-a)

        return grad_x, grad_y


def adaptive_gradient(U, V, dx, dy):
    """
    Calculate the gradients of vector field (U, V) adaptively using central or one-sided differences.
    
    U, V: 2D numpy arrays representing the x and y components of the vector field.
    
    Returns:
    dU_dx, dU_dy, dV_dx, dV_dy: 2D numpy arrays of the same shape as U, V representing the gradients.
    """
    # Get the shape of the input arrays
    ny, nx = U.shape
    
    # Initialize gradient arrays
    dU_dx = np.zeros((ny, nx))
    dU_dy = np.zeros((ny, nx))
    dV_dx = np.zeros((ny, nx))
    dV_dy = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            # Calculate dU/dx
            if j > 0 and j < nx - 1 and U[i, j-1] != 0 and U[i, j+1] != 0:
                dU_dx[i, j] = (U[i, j+1] - U[i, j-1]) / (2*dx)
            elif j < nx - 1 and U[i, j] != 0 and U[i, j+1] != 0:
                dU_dx[i, j] = (U[i, j+1] - U[i, j]) / dx
            elif j > 0 and U[i, j] != 0 and U[i, j-1] != 0:
                dU_dx[i, j] = (U[i, j] - U[i, j-1]) / dx
            
            # Calculate dU/dy
            if i > 0 and i < ny - 1 and U[i-1, j] != 0 and U[i+1, j] != 0:
                dU_dy[i, j] = (U[i+1, j] - U[i-1, j]) / (2*dy)
            elif i < ny - 1 and U[i, j] != 0 and U[i+1, j] != 0:
                dU_dy[i, j] = (U[i+1, j] - U[i, j]) / dy
            elif i > 0 and U[i, j] != 0 and U[i-1, j] != 0:
                dU_dy[i, j] = (U[i, j] - U[i-1, j]) / dy
            
            # Calculate dV/dx
            if j > 0 and j < nx - 1 and V[i, j-1] != 0 and V[i, j+1] != 0:
                dV_dx[i, j] = (V[i, j+1] - V[i, j-1]) / (2 * dx)
            elif j < nx - 1 and V[i, j] != 0 and V[i, j+1] != 0:
                dV_dx[i, j] = (V[i, j+1] - V[i, j]) /  dx
            elif j > 0 and V[i, j] != 0 and V[i, j-1] != 0:
                dV_dx[i, j] = (V[i, j] - V[i, j-1]) / dx
            
            # Calculate dV/dy
            if i > 0 and i < ny - 1 and V[i-1, j] != 0 and V[i+1, j] != 0:
                dV_dy[i, j] = (V[i+1, j] - V[i-1, j]) / (2 * dy)
            elif i < ny - 1 and V[i, j] != 0 and V[i+1, j] != 0:
                dV_dy[i, j] = (V[i+1, j] - V[i, j]) / dy
            elif i > 0 and V[i, j] != 0 and V[i-1, j] != 0:
                dV_dy[i, j] = (V[i, j] - V[i-1, j]) / dy

    return dU_dx, dU_dy, dV_dx, dV_dy



#----------------------------------------
# Create dense velocity and density data
#----------------------------------------

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


def velocity_interpolation(Dataset, gridspace, density_folder, velocity_folder, sigma=0.0):
        """
        Uses the calculated velocity vector fields to interpolate in between values and match them to a grid

        Parameters
        ----------
        Dataset : str
            'Cut' or 'Delta_Catenin', depending on what you use
        """  
        
        # Path to the Folders
        general_path = os.path.join(dataset_dir, Dataset)
        density_path = os.path.join(general_path, density_folder)
        velocity_path = os.path.join(general_path, velocity_folder)
        data_file = os.path.join(general_path, f'{Dataset}.csv')        # Cell-tracking data (Original data)

        if not os.path.isdir(velocity_path):
                os.makedirs(velocity_path)

        if Dataset == 'Cut':
                xy_shape = [0, 720,0, 2240] # xmin xmax ymin ymax
        elif Dataset == 'Delta_Catenin':
                xy_shape = [25, 1520, 25, 1520]
        else:
                raise Exception("Select a correct dataset, either 'Cut' or 'Delta_Catenin'.")

        t = np.loadtxt(data_file, delimiter=",", skiprows=4, usecols = (7), unpack=True)
        delta_t = np.round(np.unique(t))[1]
        time_points = np.round(np.unique(t/delta_t)).astype(int)[-1]

        create_csv(Dataset, time_points, use='Velocity_Interpolation', folder_name = velocity_folder)

        #for i in tqdm(range(0, time_points)):

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
                grid_x, grid_y = np.meshgrid(np.linspace(xy_shape[0], xy_shape[1], int((xy_shape[1]-xy_shape[0])/gridspace+1)), np.linspace(xy_shape[3], xy_shape[2], int((xy_shape[3]-xy_shape[2])/gridspace+1)))

                # Interpolation
                coordinates = np.array([x_cells, y_cells]).T
                grid_u = griddata(coordinates, u_cells, (grid_x, grid_y), method='cubic')
                grid_v = griddata(coordinates, v_cells, (grid_x, grid_y), method='cubic')


                #grid_u[-1,:] = 0
                #grid_u[0, :] = 0
                #grid_u[:,-1] = 0
                #grid_u[:, 0] = 0

                #grid_v[-1,:] = 0
                #grid_v[0, :] = 0
                #grid_v[:,-1] = 0
                #grid_v[:, 0] = 0

                # Gaussian Filter
                grid_u = gaussian_filter(grid_u, sigma=sigma)
                grid_v = gaussian_filter(grid_v, sigma=sigma)

                # Saving data
                grid_x = grid_x.ravel() 
                grid_y = grid_y.ravel() 
                grid_u = grid_u.ravel() 
                grid_v = grid_v.ravel()    

                data = np.column_stack((grid_x, grid_y, grid_u, grid_v))

                with open(velocity_data, 'a', newline='') as csvfile:
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

 
#--------------
# Plotting
#--------------

def density_plot(Dataset, quantity, xedges, yedges, title, save=False, plot_folder=''):
        """
        Plot displaying a matrix with imshow

        Parameters
        ----------
        quantity : array
            Data to be displayed, should be a numpy array matrix
        title : str
           Name of the plot
        xedges: array
                extent of the data in x
        yedges: array
                extent of the data in y
        """

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, title=title[:-6])

        if np.nanmin(quantity) < 0:
            cmap = plt.get_cmap('seismic')
            norm = TwoSlopeNorm(vmin=np.min(quantity), vcenter=0, vmax=np.max(quantity))    # Center the colors such that white = 0.0
            pos = ax.imshow(quantity.T, interpolation=None, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],norm=norm, cmap=cmap)
            #pos = ax.imshow(quantity.T, interpolation=None, extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],norm=norm, cmap=cmap)

        else:  
            cmap = plt.get_cmap('Oranges')
            pos = ax.imshow(quantity.T, interpolation=None, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap)

        #fig.colorbar(pos, ax=ax)
        if save == True:
                if plot_folder:
                        path = os.path.join(plot_dir(Dataset, plot_folder), f'{Dataset}_{title}.png')
                else:
                        path = os.path.join(result_dir(Dataset), f'{Dataset}_{title}.png')
                plt.savefig(path)
                plt.close(fig)
        else:
                plt.savefig('Test TCD.pdf')
                plt.show()


def density_logplot(Dataset, quantity, xedges, yedges, title, save=False, plot_folder=''):
        """
        Plot displaying a matrix with imshow in a logscale, previously negative values displayed in blue, positive ones in blue.

        Parameters
        ----------
        quantity : array
            Data to be displayed, should be a numpy array matrix
        title : str
           Name of the plot
        xedges: array
                extent of the data in x
        yedges: array
                extent of the data in y
        """


        # Separate positive and negative values
        positive_data = np.where(quantity > 0, quantity, np.nan)
        negative_data = np.where(quantity < 0, quantity, np.nan)

        # Apply logarithmic scaling
        log_positive_data = np.log(positive_data)
        log_negative_data = np.log(-negative_data)

        # Define custom colormap
        class MidpointNormalize(mcolors.Normalize):
                def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
                        self.midpoint = midpoint
                        super().__init__(vmin, vmax, clip)

                def __call__(self, value, clip=None):
                        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

        # Create two separate colormaps
        cmap_positive = plt.cm.Reds
        cmap_negative = plt.cm.Blues

        # Normalize log values
        norm = MidpointNormalize(vmin=np.nanmin(log_negative_data), vmax=np.nanmax(log_positive_data), midpoint=0)

        # Create plot
        fig, ax = plt.subplots()

        # Plot positive values
        pos_img = ax.imshow(log_positive_data,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap_positive, norm=norm)
        # Plot negative values
        neg_img = ax.imshow(log_negative_data,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap_negative, norm=norm)

        plt.title(f't = {title}')

        if save == True:
                if plot_folder:
                        path = os.path.join(plot_dir(Dataset, plot_folder), f'{Dataset}_{title}.png')
                else:
                        path = os.path.join(result_dir(Dataset), f'{Dataset}_{title}.png')
                plt.savefig(path)
                plt.close(fig)
        else:
                plt.savefig('Test log TCD.pdf')
                plt.show()


def density_tracking(Dataset, quantity, xedges, yedges, title, save=False, plot_folder=''):
        """
        Plot displaying a matrix with imshow

        Parameters
        ----------
        quantity : array
            Data to be displayed, should be a numpy array matrix
        title : str
           Name of the plot
        xedges: array
                extent of the data in x
        yedges: array
                extent of the data in y
        """
        
        data = np.abs(quantity)
        #data[data == 0] = 1e-50         # Avoid 0 in logarithm
        #data = np.log10(np.abs(data))     # Use logarithm for easier detection

        # Preprocess the image (optional)
        blurred_image = gaussian_filter(data, sigma=2)
        #blurred_image = data
        # Apply thresholding to create a binary image
        #threshold_value = threshold_otsu(blurred_image)
        threshold_value = 0.9*np.max(blurred_image)

        binary_image = blurred_image > threshold_value

        # Label connected components
        labeled_image, num_features = label(binary_image)

        # Find the center of mass of each particle
        centroids = center_of_mass(binary_image, labeled_image, range(1, num_features + 1))

        #print(np.min(data), np.max(data))

        #  Visualize the results
        plt.figure(figsize=(8, 8))
        cmap = plt.get_cmap('seismic')

        norm = TwoSlopeNorm(vmin=np.min(quantity), vcenter=0, vmax=np.max(quantity))    # Center the colors such that white = 0.0
        plt.imshow(quantity, interpolation=None ,norm=norm, cmap=cmap)
        plt.scatter([c[1] for c in centroids], [c[0] for c in centroids], color='red', s=30)
        plt.title(title)

        if save == True:
                if plot_folder:
                        path = os.path.join(plot_dir(Dataset, plot_folder), f'{Dataset}_{title}.png')
                else:
                        path = os.path.join(result_dir(Dataset), f'{Dataset}_{title}.png')
                plt.savefig(path)
                plt.close()

                x_defect = [c[0] for c in centroids]
                y_defect = [c[1] for c in centroids]
        else:
                plt.savefig('Test log TCD.pdf')
                plt.show()
        return x_defect, y_defect


def quiver_plot(x, y, u, v, Dataset, title, step_x=1, step_y=1, threshold=0.0, normalized=True, save=False, plot_folder=''):
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
        step: int
                Reduces amount of arrows by the defined stepsize
        threshold: float
                Hides arrows under a certain threshold (not really useful)
        normalized: bool
                Arrows have the same length, cannot be combined with threshold
        """

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


        plt.figure(figsize=(45,15))

        if threshold != 0.0:
                mask = magnitude >= threshold
                x = x[mask]
                y = y[mask]
                u = u[mask]
                v = v[mask]
                magnitude_filtered = magnitude[mask]
                plt.quiver(x, y, u, v, magnitude_filtered, cmap='viridis')


        elif normalized == True:
                u = u / magnitude
                v = v / magnitude

                #u = u*30
                #v = v*30
                # For color-labeled arrows depending on amplitude
                #threshold = 0.02
                #mask = magnitude < threshold

                # Plot arrows below the threshold in red
                #plt.quiver(x[mask], y[mask], u[mask], v[mask], angles='xy', color=frankfurt, scale=0.05, scale_units='xy')

                # Plot arrows above the threshold in blue
                #plt.quiver(x[~mask], y[~mask], u[~mask], v[~mask], angles='xy', color=hamburg, scale=0.05, scale_units='xy')

                #Non-colored plot
                if step_x == 5:
                        scale = 60
                elif step_x == 1:
                        scale = 200
                elif step_x == 3:
                        scale = 105
                elif step_x == 30:
                        scale = 105
                else:
                        raise ValueError('Step_size not know, add new scale value in quiver_plot')
                        
                plt.quiver(x, y, u, v, np.arctan2(v, u), cmap='hsv', scale=scale, minshaft=2)
        else:
                plt.quiver(x, y, u, v)


        #plt.title(title)
        if save==True: 
                if plot_folder:
                        path = os.path.join(plot_dir(Dataset, plot_folder), f'{Dataset}_{title}.png')
                else:
                        path = os.path.join(result_dir(Dataset), f'{Dataset}_{title}.png')
                plt.savefig(path)
                plt.close()
        else:
                plt.savefig('Test.pdf')
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

def velocity_plots(Dataset, data_folder, save_folder, t_lim, step_x, step_y, part):
        """
        Creates Images and a Video of the velocity plots

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

        # Create quiver plot 
        general_velocity_path = r'C:\Users\User\Desktop\Physikstudium\Masterstudium\Summer Project\Datasets\Cut'
        data_folder = os.path.join(general_velocity_path, data_folder)

        if part == 1:
                start = 250
                end = 251
        if part == 2:
                start = 250
                end = 260

        for i in tqdm(range(start, end), desc='Calculating images 1/2, take a coffe break'):
                velocity_path = os.path.join(data_folder, f'Cut_{10000+i}.csv')
                x, y, u, v = np.loadtxt(velocity_path, delimiter=",", skiprows=1, unpack=True)

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

                quiver_plot(y, x, v ,u,  'Cut', f'{10000+i}', step_x=step_x, step_y=step_y, save=False, plot_folder=save_folder)

        if part == 2:
                # Create Movie
                output_video = f'{Dataset}_{save_folder}.mp4'
                frame_rate = 10  # Frames per second
                save_path = plot_dir(Dataset, save_folder)
                create_video_from_images(save_path, output_video, frame_rate)


def experimental_charge_velocity(Dataset,velocity_folder, TCD_folder, plot_folder, part):
        """
        Stupid function to calculate topological charge densities, write these into a file, plot them and make a video
        Since my laptop cannot handle to many plots the function has to be called upon twice, part=1 and part=2...

        Parameters
        ----------
        Dataset: str
                'Cut' or 'Delta_Catenin'  
        velocity_folder: str
                Where the interpolated velocities are stored
        plot_folder: str
                Where you want the images to be stored, folder does not have to exist previously
        part: int
                Either 1 or 2...      
        """

        general_path = os.path.join(dataset_dir, Dataset)
        save_TCD_path = os.path.join(general_path, TCD_folder)
        defect_save_folder  = os.path.join(general_path, 'Defect_tracked')

        if not os.path.isdir(save_TCD_path):
                os.makedirs(save_TCD_path)
        if not os.path.isdir(defect_save_folder):
                os.makedirs(defect_save_folder)

        time_points = len(fnmatch.filter(os.listdir(os.path.join(general_path, velocity_folder)), '*.csv'))

        # Calculates topological charge density and writes it into files
        if part == 0:
                for i in tqdm(range(0, 379), desc="Calculating Topological Charge density while you do nothing you lazy papaya"):
                        file = f'{Dataset}_{10000+i}.csv'
                        save_path = os.path.join(save_TCD_path, file)
                        file_velocity = os.path.join(velocity_folder, file)
                        TC_without_density(Dataset, file_velocity=file_velocity, save_path=save_path)

                print('Calculation of topological quantities done')

        # Plots images of the Topological charge density and saves them into a folder
        if part == 1:
                start = 0
                end = 356
        elif part == 2:
                start = 356
                end = 379
        else:
                start = 0
                end = 0

        for i in tqdm(range(start, end), desc='Generating Topological charge density images'):
                file = f'{Dataset}_{10000+i}.csv'
                TCD_file = os.path.join(save_TCD_path, file)

                rho = np.loadtxt(TCD_file, delimiter=",", skiprows=1, unpack=True, usecols=(2))
                rho = rho.reshape(223, 71) # Gridspacing = 10

                #density_plot('Cut', rho.T, [0,720], [0, 2240], f'Cut_TCD_{10000+i}', save=False, plot_folder=plot_folder)
                #density_logplot('Cut', rho, [0,720], [0, 2240], f'Cut_TCD_{10000+i}', save=False, plot_folder=plot_folder)
                x_defect, y_defect = density_tracking('Cut', rho, [0,720], [0, 2240], f'Cut_TCD_{10000+i}', save=True, plot_folder=plot_folder)

                # Save defect position
                defect_file = os.path.join(defect_save_folder, f"Cut_{10000+i}.csv")
                data = np.column_stack((x_defect, y_defect))
                with open(defect_file, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['x','y'])
                        writer.writerows(data)


        # Uses the images from the folder to create a video
        if part == 2:
                # Create Movie
                output_video = f'{Dataset}_{plot_folder}.mp4'
                frame_rate = 10  # Frames per second
                image_files = plot_dir(Dataset, plot_folder)

                create_video_from_images(image_files, output_video, frame_rate)
        
        return 0


def iterate_theo_charge(Dataset):
        """
        Calculates the theoretical topological charge velocities and saves them in a file
        """
        dataset_folder = os.path.join(dataset_dir, Dataset)
        velocity_folder = os.path.join(dataset_folder, 'Cut_Velocity_3')
        save_pre_file = os.path.join(dataset_folder, 'Vector_Charges')

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

                charge_x, charge_y = theoretical_charge_velocity(u, v, u_prev, v_prev, u_next, v_next, dx, dy, shape=(225,73))
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


def charges_quiver(Dataset,title, plot_folder, Velocities_file, Defect_file, step_x, step_y, normalized=True, save=True):
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

        sigma = 3
        #u = gaussian_filter(u, sigma=sigma)
        #v = gaussian_filter(v, sigma=sigma)



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


        #plt.title(title)
        if save==True: 
                path = os.path.join(plot_dir(Dataset, plot_folder), f'{Dataset}_{title}.png')
                plt.savefig(path)
                plt.close()
        else:
                plt.savefig('Test.pdf')
                plt.show()


def iterate_charge_plots(Dataset, t_lim, step_x, step_y, part):
        """
        Creates Images and a Video of the velocity plots

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
        vector_charges_folder = os.path.join(general_path, 'Vector_Charges')
        defect_folder = os.path.join(general_path, 'Defect_tracked')
        plot_folder = os.path.join(result_dir(Dataset), 'Non_smoothed tracking velocity')

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
                output_video = f'{Dataset}_non_smoothed.mp4'
                frame_rate = 10  # Frames per second
                save_path = plot_dir(Dataset, plot_folder)
                create_video_from_images(save_path, output_video, frame_rate)