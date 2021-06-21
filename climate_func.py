#funciones.py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
from scipy import signal
import scipy as sp
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath

#Funciones main
def matrix(dato):
    M = np.zeros([int(len(dato.lat)*len(dato.lon)), len(dato.time)])
    cont = 0
    for i in range(len(dato.lat)):
        for j in range(len(dato.lon)):
            if (dato.lat[i].values == 0) or (dato.lat[i].values == 90) or (dato.lat[i].values == -90):
                coslat = 0
                time_evolution_ij = np.nan_to_num(dato.values[:,i,j],0) 
                M[cont] =  signal.detrend(time_evolution_ij) * np.sqrt(coslat)
                cont += 1
            else:
                coslat = np.cos(np.deg2rad(dato.lat[i].values))
                time_evolution_ij = np.nan_to_num(dato.values[:,i,j],0) 
                M[cont] =  signal.detrend(time_evolution_ij) * np.sqrt(coslat)
                cont += 1
    return M

def VT_to_pattern(data,vt,n):
    eof_pattern = np.zeros((len(data.lat),len(data.lon)))
    cont = 0
    for i in range(len(data.lat)):
        for j in range(len(data.lon)):
            eof_pattern[i,j] = vt[n-1,cont]
            cont += 1

    return eof_pattern

def pattern_maps(data,vt):
    eof_pattern = np.zeros((len(data.lat),len(data.lon)))
    cont = 0
    for i in range(len(data.lat)):
        for j in range(len(data.lon)):
            eof_pattern[i,j] = vt[cont]
            cont += 1

    return eof_pattern

#Remove climatology
def anomalias(dato):
    dato_anom = dato.groupby('time.month') - dato.groupby('time.month').mean('time')

    return dato_anom

#Select an area
def area(dato,lon_w, lon_e, lat_s, lat_n):
    average = dato.sel(lat=slice(lat_n,lat_s)).sel(lon=slice(lon_w,lon_e))
    return average

#Select a period
def climate(dato,year_1,year_2):
    climate = dato.sel(time=slice(year_1,year_2))
    return climate

def plot_fig(axs,lat,lon,dat,data_crs,clevels,cmap):
    img = axs.contourf(lon, lat, dat,clevels,transform=data_crs,cmap=cmap,extend='both')
    axs.add_feature(cartopy.feature.COASTLINE)
    axs.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    axs.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    #axs.set_title(modelos[i-1],fontsize=18)
    return img

#SoutherHemisphere Stereographic
def plot_stereo(axs,lat,lon,dat,clevels,cmap):
    projection = ccrs.SouthPolarStereo(central_longitude=300)
    data_crs = ccrs.PlateCarree()
    axs.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    axs.set_boundary(circle, transform=axs.transAxes)
    im = axs.contourf(lon, lat, dat,clevels,transform=data_crs,cmap=cmap,extend='both')
    #cnt=axs.contour(lonh,lath, climU850,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    #levels = [vb_eescp.min(),0.05,vb_eescp.max()]
    #axs.contourf(lon_, lat, pvals,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    axs.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    axs.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    axs.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    axs.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    return axs, im

def cross_year_season(month):
    return (month >= 12) | (month <= 2)
    #return (month >= 9) & (month <= 11)
    
def plot_basic_map(mapa1,mapa2,lat,lon,title):
    fig = plt.figure(figsize=(20,10))
    gs = fig.add_gridspec(3, 4)
    ax1 = fig.add_subplot(gs[0:2, :2], projection=ccrs.PlateCarree(180))
    ax2 = fig.add_subplot(gs[0:2, 2:], projection=ccrs.Robinson(180))
    clevels1 = np.arange(-1,1.1,0.1)
    clevels2 = np.arange(-1,1.1,0.1)
    im1 = plot_fig(ax1,lat, lon, mapa1/np.max(mapa1),ccrs.PlateCarree(),clevels1,cmap='PuOr')
    cbar = plt.colorbar(im1, orientation='vertical')
    im2 = plot_fig(ax2,lat, lon, mapa2/np.max(mapa2),ccrs.PlateCarree(),clevels2,cmap='RdBu_r')
    cbar = plt.colorbar(im2, orientation='vertical')
    fig.suptitle(title)
           
    return fig
