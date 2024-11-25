import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import holoviews as hv
import geoviews as gv
import geoviews.feature as gf
from cartopy import crs as ccrs
import panel as pn
import datashader as ds
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import ipywidgets as widgets
import re
from sklearn.cluster import KMeans
import rasterio as rio
import hvplot.xarray
import rioxarray
from scipy.linalg import norm, eig
import numpy.ma as ma
from cartopy import crs as ccrs
import panel as pn
from sklearn.preprocessing import normalize

def extract_values_from_line(line):
    """Extracts float values from a line containing a list of values within curly braces."""
    values_str = re.search(r"\{(.*?)\}", line).group(1)
    values = [float(value.strip()) for value in values_str.split(',')]
    return values

def parse_hdr_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    wavelengths = []
    fwhms = []

    for line in lines:
        if "wavelength =" in line:
            wavelengths = extract_values_from_line(line)
        elif "fwhm =" in line:
            fwhms = extract_values_from_line(line)

    return wavelengths, fwhms

def linear_stretch(data, lower_percentile=2, upper_percentile=98):
    # Compute percentiles for each band
    lower_bound = data.quantile(lower_percentile/100, dim=['x', 'y'])
    upper_bound = data.quantile(upper_percentile/100, dim=['x', 'y'])
    
    # Apply linear stretch
    stretched = (data - lower_bound) / (upper_bound - lower_bound)
    stretched = stretched.clip(0, 1)  # Ensure the values are within [0, 1]
    
    return stretched

def brighten(rgb, f=2):
    brightness_factor = f # Adjust this factor to control brightness
    rgb_normalized = (rgb / np.nanmax(rgb)) * brightness_factor
    rgb_normalized = rgb_normalized.clip(min=0, max=1)  # Ensure values are within 0-1 range
    return rgb_normalized

def spec_viz(data, unc, transform='linear_stretch'):

    # Plotting
    rgb = data.sel(wavelength=[670, 560, 470], method='nearest')
    if transform == 'brighten':
        rgb_norm = brighten(rgb, f=1.5)
    elif transform == 'linear_stretch':
        rgb_norm = linear_stretch(rgb, lower_percentile=2, upper_percentile=98)
    else:
        # error
        print('Invalid transform: choose from "brighten" or "linear_stretch"')
        return None
    
    map = rgb_norm.hvplot.rgb(
            x='x', y='y',  bands='wavelength',
            aspect='equal',  # rasterize=True,
            # crs=ccrs.PlateCarree(),
            width=700,
            height=700,
        )

    # Stream of X and Y positional data
    posxy = hv.streams.PointerXY(source=map, x=-79.910, y=9.415)
    clickxy = hv.streams.Tap(source=map, x=-79.937, y=9.320)

    # Function to build a new spectral plot based on mouse hover positional information retrieved from the RGB image using our full reflectance dataset
    def point_spectra(x, y):
        pointed = data.sel(x=x, y=y, method='nearest')
        uncert = unc.sel(x=x, y=y, method='nearest')

        p1 = pointed.hvplot.line(y='reflectance', x='wavelength', color='red', frame_width=500,)
        p2 = pointed.hvplot.scatter(y='reflectance', x='wavelength').opts(color='red', size=20, marker='x',)
        p3 = hv.Area((pointed.wavelength, pointed.data - uncert.data, pointed.data + uncert.data), vdims=['y1','y2']).opts(alpha=0.3, color='red')
        
        ps = p1 * p2 * p3
        return ps

    # Function to build spectral plot of clicked location to show on hover stream plot
    def click_spectra(x, y):
        clicked = data.sel(x=x, y=y, method='nearest')
        uncert = unc.sel(x=x, y=y, method='nearest')

        p1 = clicked.hvplot.line(y='reflectance', x='wavelength', color='black', frame_width=500).opts(
            title=f'Latitude = {clicked.y.values.round(3)}, Longitude = {clicked.x.values.round(3)}',)
        p2 = clicked.hvplot.scatter(y='reflectance', x='wavelength', ).opts(color='black', size=20, marker='x',)
        p3 = hv.Area((clicked.wavelength, clicked.data - uncert.data, clicked.data + uncert.data), vdims=['y1','y2']).opts(alpha=0.3, color='black')
        
        ps = p1 * p2 * p3
        return ps

    # Define the Dynamic Maps
    point_dmap = hv.DynamicMap(point_spectra, streams=[posxy],)
    click_dmap = hv.DynamicMap(click_spectra, streams=[clickxy],)

    # Plot the Map and Dynamic Map side by side
    # return (map + click_dmap * point_dmap)
    pn.serve(map + click_dmap * point_dmap)

    return rgb_norm


def elbow (data):
    # elbow method to define # clusters
    # requires visual inspection of plot
    # data = data.values(data)
    K = range(1,8)
    dist = []
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        dist.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(K,dist,'ko-')


def kmeans_clust (data,k):
    # k means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(data)
    labels = kmeans.predict(data)
    #centroids = kmeans.cluster_centers_
    return labels

# Define a function to clip and remove outliers
def preprocess_vi(vi, clip_min, clip_max):
    vi_clipped = np.clip(vi, clip_min, clip_max)
    q1 = np.nanpercentile(vi_clipped, 25)
    q3 = np.nanpercentile(vi_clipped, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    vi_cleaned = np.where((vi_clipped > lower_bound) & (vi_clipped < upper_bound), vi_clipped, np.nan)
    return vi_cleaned