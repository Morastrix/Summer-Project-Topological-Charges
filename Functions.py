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

from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm


# Path to the script and to the folder for the results
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/C07/')
dataset_dir = r'C:\Users\User\Desktop\Physikstudium\Masterstudium\Summer Project\Datasets'


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


# Colorcoding
lightgreen = '#1DA64A'
darkgreen = '#1B7340'
hamburg = '#549670'
blue = '#90D3ED'
red = '#f5333f'
frankfurt = '#ffc000'


def load_data(file, Dataset, measurement_filtering=False, filtering=None):
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
                uniform_density = True 

        elif Dataset == 'Cut' or Dataset == 'C05' or Dataset == 'C06' or Dataset == 'Delta_Catenin':
                file_velocity = os.path.join(Dataset, f'{Dataset}_Velocity')
                file_velocity = os.path.join(file_velocity, file)
                file_density = os.path.join(Dataset, f'{Dataset}_Velocity')
                file_density = os.path.join(file_velocity, file)
                uniform_density = False

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
                spread = 3.0
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


def topological_quantities(file, Dataset, measurement_filtering=False, filtering='uniform', save_path=''):
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


def theoretical_charge_velocity(file_prev, file, file_next, filtering='uniform'):
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
        filtering: str, optional
                Filter used to spread out cells over more than 1 pixel, either None, 'uniform', or 'gaussian' 
        """

        time_interval = 1

        # Load in data from the following and previous image
        Binned_cells_next, u_middle_next, v_middle_next = load_data(file_next, filtering=filtering)[0:3]
        Binned_cells_prev, u_middle_prev, v_middle_prev = load_data(file_prev, filtering=filtering)[0:3]
        Binned_cells, u_middle, v_middle, x, y = load_data(file, filtering=filtering)[0:5]


        # Spatial derivatives of the actual image
        dx = np.unique(x)[1]-np.unique(x)[0]
        dy = np.unique(y)[1]-np.unique(y)[0] 

        flux_u = np.multiply(Binned_cells, u_middle)
        flux_v = np.multiply(Binned_cells, v_middle)

        dx_flux_u = np.gradient(flux_u, dx, axis=1)
        dy_flux_u = np.gradient(flux_u, dy, axis=0)
        dx_flux_v = np.gradient(flux_v, dx, axis=1)
        dy_flux_v = np.gradient(flux_v, dy, axis=0)


        # Fluxes of surrounding images to calculate the time derivative
        flux_u_next = np.multiply(Binned_cells_next, u_middle_next)
        flux_v_next = np.multiply(Binned_cells_next, v_middle_next)
        flux_u_prev = np.multiply(Binned_cells_prev, u_middle_prev)
        flux_v_prev = np.multiply(Binned_cells_prev, v_middle_prev)

        # TODO: Derivative is missing at the edges, so first frame and last frame
        # TODO: Not super accurate derivative and quite slow for multiple images
        # Calculate the time derivative of the flux using the symmetric difference quotient
        dt_flux_u = (flux_u_next-flux_u_prev)/(2*time_interval)          
        dt_flux_v = (flux_v_next-flux_v_prev)/(2*time_interval)

        # TODO: Issue with the division by k, since k is 0 at many positions
        k = np.multiply(dx_flux_u, dy_flux_v)-np.multiply(dx_flux_v, dy_flux_u)
        charge_vx = -2*(dt_flux_u * dy_flux_v - dt_flux_v * dy_flux_u) / k 
        charge_vy = -2*(dt_flux_v * dx_flux_u - dt_flux_u * dx_flux_v) / k 

        return charge_vx, charge_vy


def experimental_charge_velocity(Dataset):

        general_path = os.path.join(dataset_dir, Dataset)
        save_path = os.path.join(general_path, f'{Dataset}_TCD')

        time_points = len(fnmatch.filter(os.listdir(os.path.join(general_path, f'{Dataset}_Velocity')), '*.csv'))
        print(time_points)

        for i in tqdm(range(0, time_points), desc="Calculating Topological Charge density while you do nothing you lazy pampaya"):
                file = f'{Dataset}_{10000+i}.csv'
                save_name = f'{Dataset}_{10000+i}.csv'
                save_file = os.path.join(save_path, save_name)
                topological_quantities(file, Dataset, measurement_filtering=False, filtering='uniform', save_path=save_file)

        
#--------------
# Plotting
#--------------

def density_plot(quantity, xedges, yedges, title, save=False):
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
            pos = ax.imshow(quantity.T, interpolation=None, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, norm=norm)
        else:  
            cmap = plt.get_cmap('Oranges')
            pos = ax.imshow(quantity.T, interpolation=None, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap)

        #fig.colorbar(pos, ax=ax)
        if save == True:
                plt.savefig(f'{results_dir + title}.png')
                plt.close()
        else:
            plt.show()


def quiver_plot(x, y, u, v, title, step=1, threshold=0.0, normalized=False, save=False):
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
                Hides arrows under a certain threshold
        normalized: bool
                Arrows have the same length, cannot be combined with threshold
        """
        
        # Reduce arrows
        x = x[::step,::step]
        y = y[::step,::step]
        u = u[::step,::step]
        v = v[::step,::step]

        magnitude = np.sqrt(u**2 + v**2)

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

                # For color-labeled arrows depending on amplitude
                threshold = 0.05
                mask = magnitude < threshold

                # Plot arrows below the threshold in red
                plt.quiver(x[mask], y[mask], u[mask], v[mask], color=frankfurt, scale=0.2/step, scale_units='xy')

                # Plot arrows above the threshold in blue
                plt.quiver(x[~mask], y[~mask], u[~mask], v[~mask], color=hamburg, scale=0.2/step, scale_units='xy')

                # Non-colored plot
                #plt.quiver(x, y, u, v, scale=0.025, scale_units='xy')
        else:
                plt.quiver(x, y, u, v)

        plt.title(title)
        if save==True: 
                plt.savefig(f'{results_dir + title}.pdf')
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


#----------------------------------------
# Create dense velocity and density data
#----------------------------------------

def measurement_v_filter(x_cells, y_cells, u_cells, v_cells):
    """
    Filter out measurements with y-values far above the others. Also includes velocity values

    Parameters
    ----------
    x_cells: array
        x-coordinates of the cells
    y_cells: array
        y-coordinates of the cells
    u_cells: array
        u-coordinates of the cells
    v_cells: array
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
        u_cells = np.delete(u_cells, indices_delete)
        v_cells = np.delete(v_cells, indices_delete) 
    return x_cells, y_cells, u_cells, v_cells


def measurement_filter(x_cells, y_cells):
    """
    Filter out measurements with y-values far above the others. Without velocity values

    Parameters
    ----------
    x_cells: array
        x-coordinates of the cells
    y_cells: array
        y-coordinates of the cells
    u_cells: array
        u-coordinates of the cells
    v_cells: array
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
    return x_cells, y_cells


def create_csv(Dataset, time_points, use='', header=[]):
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
                Folder = os.path.join(path, f'{Dataset}_Velocity')
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


def velocity_csv(Dataset):
        """
        Uses the given data consisting of track_ID, x, y and t to calculate individual cell velocities and store them in a csv file 
        according to the correct time. (Used for Cut dataset)

        Parameters
        ----------
        Dataset : str
            'Cut' or 'Delta_Catenin', depending on what you use
        """        

        general_path = os.path.join(dataset_dir, Dataset)
        densiy_path = os.path.join(general_path, f'{Dataset}_Density')
        file_path = os.path.join(general_path, 'Chunk_Data')

        track_ID, x, y, t =  np.loadtxt(file_path, delimiter=",", skiprows=4, usecols = (2,4,5,7), unpack=True)
        delta_t = np.round(np.unique(t))[1]
        time_points = np.round(np.unique(t/delta_t)).astype(int)[-1]

        create_csv('Delta_Catenin', time_points, use='Velocity', header=['x', 'y', 'u', 'v']) # Creates csv files for each timepoint

        # Iterates through all track_IDs and write x, y, u, v in a csv file according to their timepoint.
        for elem in tqdm(np.unique(track_ID), desc="Progress"): # Selects individual cells
                # Finds indices with the selected track_ID
                indices = np.where(track_ID == elem)
                x_elem = x[indices]
                y_elem = y[indices]
                t_elem = t[indices]

                sort_index = np.argsort(t_elem)

                x_elem = x_elem[sort_index]
                y_elem = y_elem[sort_index]
                t_elem = t_elem[sort_index]

                u = np.gradient(x_elem, t_elem)
                v = np.gradient(y_elem, t_elem)

                file_number = np.round(np.unique(t_elem/delta_t)).astype(int)  
                # Go through all times, find the corresponding csv file given by 'time' and write x,y,u,v data in this file

                for idx, time in enumerate(file_number):
                        csv_name = f'{Dataset}_{int(10000+time)}.csv'
                        csv_file_path = os.path.join(densiy_path, csv_name)
                        with open(csv_file_path, 'a', newline='') as csvfile:
                                # Create a csv.writer object
                                fieldnames = ['x', 'y', 'u', 'v']
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writerow({'x': x_elem[idx], 'y': y_elem[idx], 'u': u[idx], 'v' : v[idx]})


def velocity_interpolation(Dataset):
        """
        Uses the calculated velocity vector fields to interpolate in between values and match them to a grid

        Parameters
        ----------
        Dataset : str
            'Cut' or 'Delta_Catenin', depending on what you use
        """  
        
        # Path to the Folders
        general_path = os.path.join(dataset_dir, Dataset)
        density_folder = os.path.join(general_path, f'{Dataset}_Density')
        velocity_folder = os.path.join(general_path, f'{Dataset}_Velocity')
        data_file = os.path.join(general_path, 'Chunk_Data.csv')

        if Dataset == 'Cut':
                xy_shape = [0, 720,0, 2245] # xmin xmax ymin ymax
        elif Dataset == 'Delta_Catenin':
                xy_shape = [25, 1520, 25, 1520]
        else:
                raise Exception("Select a correct dataset, either 'Cut' or 'Delta_Catenin'.")

        t = np.loadtxt(data_file, delimiter=",", skiprows=4, usecols = (7), unpack=True)
        delta_t = np.round(np.unique(t))[1]
        time_points = np.round(np.unique(t/delta_t)).astype(int)[-1]

        create_csv(Dataset, time_points, use='Velocity_Interpolation')

        for i in tqdm(range(0, time_points)):
                filename = f'{Dataset}_{10000+int(i)}.csv'
                density_data = os.path.join(density_folder, filename)
                velocity_data = os.path.join(velocity_folder, filename)

                x_cells, y_cells, u_cells, v_cells = np.loadtxt(density_data, delimiter=",", skiprows=1, unpack=True)

                # Filter out NaN values in the u & v data 
                valid_mask = ~np.isnan(u_cells) & ~np.isnan(v_cells)
                x_cells = x_cells[valid_mask]
                y_cells = y_cells[valid_mask]
                u_cells = u_cells[valid_mask]
                v_cells = v_cells[valid_mask]

                # Filter out measurement values far off the rest
                x_cells, y_cells, u_cells, v_cells = measurement_v_filter(x_cells, y_cells, u_cells, v_cells)                

                # Create the grid (yes it looks fcking weird, but thats how the other data looks like) 
                grid_x, grid_y = np.meshgrid(np.linspace(xy_shape[0], xy_shape[1], int((xy_shape[1]-xy_shape[0])/5+1)), np.linspace(xy_shape[3], xy_shape[2], int((xy_shape[3]-xy_shape[2])/5+1)))

                coordinates = np.array([x_cells, y_cells]).T
                grid_u = griddata(coordinates, u_cells, (grid_x, grid_y), method='cubic')
                grid_v = griddata(coordinates, v_cells, (grid_x, grid_y), method='cubic')

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


def find_dimensions(Dataset, measurement_filtering=False):
        """
        Find the dimensions of a dataset, so max and min values of x and y
        
        Parameters
        ----------
        Dataset : str
            'Cut' or 'Delta_Catenin', depending on what you use 
        """

        x_max = 0
        y_max = 0
        x_min = 0
        y_min = 0
        File = os.path.join(dataset_dir, Dataset)
        File = os.path.join(File, 'Chunk_Data')
        t = np.loadtxt(File, delimiter=",", skiprows=4, usecols = (7), unpack=True)
        delta_t = np.round(np.unique(t))[1]
        time_points = np.round(np.unique(t/delta_t)).astype(int)[-1]
        

        for i in tqdm(range(0, time_points)): # Number of time points
                filename = f'{Dataset}_{10000+int(i)}.csv'
                origin_path = os.path.join(r'C:\Users\User\Desktop\Physikstudium\Masterstudium\Summer Project\Datasets\Delta_Catenin\Velocities', filename)       
                x, y = np.loadtxt(origin_path, delimiter=",", skiprows=1, unpack=True, usecols=(0,1))

                if measurement_filtering:
                        measurement_filter(x, y)

                if np.max(x) > x_max:
                        x_max = np.max(x)
                if np.min(x) > x_min:
                        x_min = np.min(x)
                if np.max(y) > y_max:
                        y_max = np.max(y)
                if np.min(y) > y_min:
                        y_min = np.min(y)

        print(f'Minimum x:{x_min}, Maximum x:{x_max}')
        print(f'Minimum y:{y_min}, Maximum y:{y_max}')

        return [x_min, x_max, y_min, y_max] 
