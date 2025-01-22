import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import argparse
import os
from urllib.request import urlretrieve
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



def plot_deep56ML(fits_file,vmin,vmax,save):
    with fits.open(fits_file) as hdul:
        image_data = hdul[0].data.squeeze()
    
    starting_idx = 20000 #crop
    #slide image
    rolled_image_data = np.roll(image_data, +starting_idx, axis=1)
    
    # Specify the indices for cropping
    lower_index = 15500
    upper_index = 24500
    
    #converting to uK from K, cropping
    cropped_image_data = (rolled_image_data[:,lower_index:upper_index]) * 1e6
    #Flip the data tp match RA right to left
    cropped_image_data = np.fliplr(cropped_image_data)

    height, width  = np.shape(cropped_image_data)

    # Calculate CRPIX1 and CRPIX2
    CRPIX1 = width / 2
    CRPIX2 = height / 2
    
    # Create a new WCS object
    wcs_new = WCS(naxis=2)

    #reference coordinate for the center
    ra_center = 16.0 * u.degree  
    dec_center = -2.0 * u.degree  

    resolution = 0.5 * u.arcmin.to(u.degree)

    # Set up the WCS header
    wcs_header = fits.Header()
    wcs_header['CTYPE1'] = 'RA---CAR'
    wcs_header['CUNIT1'] = 'deg'
    wcs_header['CRVAL1'] = ra_center.value
    wcs_header['CDELT1'] = resolution
    wcs_header['CRPIX1'] = CRPIX1 
    wcs_header['CTYPE2'] = 'DEC--CAR'
    wcs_header['CUNIT2'] = 'deg'
    wcs_header['CRVAL2'] = dec_center.value
    wcs_header['CDELT2'] = resolution
    wcs_header['CRPIX2'] = CRPIX2 

    # Apply the header to the WCS object
    wcs_new.wcs.set()
    wcs_new = WCS(wcs_header)
    print(wcs_new)

    #Empty values handled
    cropped_image_data[cropped_image_data == 0] = np.nan


    #figure and axis with WCS projection
    fig = plt.figure(figsize=(20, 8)) #25,6
    ax = plt.subplot(projection=wcs_new)

    #cmap = plt.get_cmap('bwr')
    #cmap.set_bad(color = 'darkgray', alpha = 1.)
    
    from matplotlib.colors import ListedColormap
    cmap_url = 'https://github.com/zonca/paperplots/raw/master/data/Planck_Parchment_RGB.txt'

    # Extract filename
    planck_cmap = cmap_url.split('/')[-1]

    if not os.path.exists(planck_cmap):
        print(f'Downloading {planck_cmap} ...')
        urlretrieve(cmap_url, planck_cmap)
    colombi1_cmap = ListedColormap(np.loadtxt(planck_cmap)/255.)
    colombi1_cmap.set_bad("gray") # color of missing pixels
    cmap = colombi1_cmap

    image = ax.pcolormesh(cropped_image_data, cmap=cmap, vmin=vmin, vmax=vmax)

    # Set the format of the tick labels to 'decimal'
    ax.coords[0].set_format_unit('deg', decimal=True)
    ax.coords[1].set_format_unit('deg', decimal=True)
    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)

    # Flip the axis#
    ax.invert_xaxis()

    ax.set_xlabel('RA (deg)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Dec (deg)', fontsize=16, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=16)

    cbar = plt.colorbar(image, ax=ax, orientation='horizontal', fraction=0.05, pad=0.15, aspect=20)
    #cbar = plt.colorbar(image, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label(r'Intensity [$\mu$K]', size=18, weight='bold')
    cbar.ax.tick_params(labelsize=14)

    plt.title('Deep 56 Field with mock 280GHz PrimeCam: \n10 dets, ~50 Hours', 
            fontsize=22, fontweight='bold')

    if save:
        plt.savefig(save)
        #plt.show()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
            description="Plot a Deep56 FITS image of ML maps",
            epilog="The vmin and vmax unit is in [uK]" 
            )
    parser.add_argument('fits_file', 
        type=str, 
        help="Path to the FITS file.")
    parser.add_argument('--vmin', type=float, 
                        default=-300, help="Minimum data value for colormap.")
    parser.add_argument('--vmax', type=float, 
                        default=300, help="Maximum data value for colormap.")
    parser.add_argument('--save', type=str, 
                        help="Path to save the output plot.")

    args = parser.parse_args()

    plot_deep56ML(args.fits_file, args.vmin, args.vmax, args.save)
    
if __name__ == '__main__':
    main()
