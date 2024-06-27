import numpy as np
from matplotlib import pyplot as plt
import Functions
import os
from tqdm import tqdm


#path = os.path.join(r'C:\Users\User\Desktop\Physikstudium\Masterstudium\Summer Project\Datasets\Cut', filename)
#Functions.velocity_interpolation(380)

#filename  = 'Cut_10055.csv'
#x,y, xedges, yedges = Functions.load_data(filename)[3:7]

#flux, rho, Binned_cells, flux_u, flux_v = Functions.topological_quantities(filename,measurement_filtering=True, filtering='uniform')
#Functions.density_plot(Binned_cells, xedges, yedges, 'YMCA')
#Functions.density_plot(rho, xedges, yedges, 'LMAO')

# C05 Dataset
#for i in tqdm(range (0,314), desc="Loading..."):
#        file = f'C05_concat-{10000+i}.csv'
#        x,y, xedges, yedges = Functions.load_data(file, 'C05')[3:7]
#        flux, rho, Binned_cells, flux_u, flux_v = Functions.topological_quantities(file, 'C05', filtering='uniform')
#        Functions.density_plot(rho, xedges, yedges,f'Topological Charge Density C05 {10000+i}', save=True)


#for i in tqdm(range (356,380), desc="Loading..."):
#        filename = f'Cut_{10000+i}.csv'
#       x,y, xedges, yedges = Functions.load_data(filename, measurement_filtering=True, filtering='uniform')[3:7]
#        flux, rho, Binned_cells, flux_u, flux_v = Functions.topological_quantities(filename, measurement_filtering=True, filtering='uniform')
#        Functions.density_plot(rho, xedges, yedges,f'Topological Charge density Cut Dataset {10000+i}', save=True)
        
# Create Movie
#image_folder = r'C:\Users\User\Desktop\Physikstudium\Masterstudium\Summer Project\Results\C05'
#output_video = 'C05_TCD.mp4'
#frame_rate = 25  # Frames per second

#Functions.create_video_from_images(image_folder, output_video, frame_rate)

#Functions.velocity_csv('Delta_Catenin')
#Functions.find_dimensions('Delta_Catenin')
#Functions.velocity_interpolation('Delta_Catenin')
Functions.experimental_charge_velocity('C07')
path = r'C:\Users\User\Desktop\Physikstudium\Masterstudium\Summer Project\Datasets\C07\C07_TCD'
for i in range(279):
    file_path = os.path.join(path, f'C07_{100000+i}')
    rho = np.loadtxt(file_path, delimiter=",", skiprows=1, unpack=True, usecols=(2))
    Functions.density_plot(rho, [10,1895], [10, 1895], f'C07_TCD_{10000+i}', save=True)