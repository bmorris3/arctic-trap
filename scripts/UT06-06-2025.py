import os
import sys
sys.path.insert(0, '../')

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time

import matplotlib
from numpy.lib.stride_tricks import sliding_window_view

from arctic_trap import (generate_master_flat_and_dark, photometry, transit_model_b,
                     PhotometryResults, PCA_light_curve, params_b)

from pathlib import Path

# Image paths
dir_path = Path("/Users/bmmorris/data/apo/Q2UW01/UT250606/")
image_paths = sorted(glob(str(dir_path / Path('obs_10s*.fits'))))
dark_30s_paths = glob(str(dir_path / Path('arctic_dark_60s.*.fits')))
night_flat_paths = glob(str(dir_path / Path('dark_sky_flat.*.fits')))
master_flat_path = 'outputs/masterflat_20250606.fits'
master_dark_path = 'outputs/masterdark_20250606.fits'

# Photometry settings
target_centroid = [401.6, 368.2]
comparison_flux_threshold = 0.05
aperture_radii = np.arange(7, 30)
centroid_stamp_half_width = 15
psf_stddev_init = 2
aperture_annulus_radius = 10
transit_parameters = params_b
star_positions = np.loadtxt('outputs/centroids-UT06-06-2025.csv')

output_path = 'outputs/trappist1_20250606.npz'
force_recompute_photometry = False

# Calculate master dark/flat:
if not os.path.exists(master_dark_path) or not os.path.exists(master_flat_path):
    print('Calculating master flat:')
    generate_master_flat_and_dark(night_flat_paths, dark_30s_paths,
                                  master_flat_path, master_dark_path)

# Do photometry:

if not os.path.exists(output_path) or force_recompute_photometry:
    print('Calculating photometry:')
    phot_results = photometry(image_paths, master_dark_path, master_flat_path,
                              star_positions,
                              aperture_radii, centroid_stamp_half_width,
                              psf_stddev_init, aperture_annulus_radius,
                              output_path)

else:
    phot_results = PhotometryResults.load(output_path)

print('Calculating PCA...')

light_curve = PCA_light_curve(phot_results, transit_parameters, plots=False,
                              plot_validation=False, buffer_time=1*u.min,
                              n_validation_frames=80, outlier_rejection=True,
                              flux_threshold=0.9)


times = Time(phot_results.times, format='jd')
window = 5
windows = sliding_window_view(light_curve, window_shape=window)
rolling_median = np.median(windows, axis=1)
formatter = matplotlib.dates.DateFormatter('%H:%M')

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(1, color='silver', ls='--')
ax.plot(times.datetime, light_curve, 'k.')
ax.plot(times.datetime[window//2:-window//2+1], rolling_median, 'r')
ax.xaxis.set_major_formatter(formatter)

# plt.plot(times.datetime, transit_model_b(phot_results.times), 'r')
ax.set(
    xlabel='Time [UT]',
    ylabel='Flux',
    ylim=[0.8, 1.20],
    title='UT06-06-2025',
)
# plt.show()
plt.savefig('outputs/UT06-06-2025.png', bbox_inches='tight', dpi=250)