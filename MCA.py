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

#Clase Maximum Covariance Analysis
class mca(object):
    def __init__(self):
        self.what_is_this = 'This is a Maximum Covariance Solver'
    
    #Hago la búsqueda standard
    def svd_solve(self,dato1,dato2,N,year1,year2):
        self.year1 = year1
        self.year2 = year2
        self.dato1 = dato1
        self.dato2 = dato2
        X = matrix(dato1)
        Y = matrix(dato2)
        self.X = X
        self.Y = Y
        #calculo matrix de covarianza
        C = (1/N) * X.dot(Y.T) 
        U, s, VT = sp.linalg.svd(C)
        #S = np.zeros((C.shape[0], C.shape[1]))
        #S[:C.shape[0], :C.shape[0]] = np.diag(s)
        self.left = U
        self.right = VT
        self.eigenvalues = s
        
    def VT_to_pattern(data,vt,n):
        pattern = np.zeros((len(data.lat),len(data.lon)))
        cont = 0
        for i in range(len(data.lat)):
            for j in range(len(data.lon)):
                eof_pattern[i,j] = vt[n-1,cont]
                cont += 1

        return pattern

    def pattern_maps(data,vt):
        pattern = np.zeros((len(data.lat),len(data.lon)))
        cont = 0
        for i in range(len(data.lat)):
            for j in range(len(data.lon)):
                pattern[i,j] = vt[cont]
                cont += 1

        return pattern

    def eof_pattern(self,n):
        left_pattern = VT_to_pattern(self.dato1,self.left,n)
        right_pattern = VT_to_pattern(self.dato2,self.right,n)
        return left_pattern, right_pattern
    
    def expansion_coef_left(self,n):
        t_x = ((self.left).T).dot(self.X) 
        return t_x[n-1]

    def expansion_coef_right(self,n):
        t_y = ((self.right).T).dot(self.Y) 
        return t_y[n-1]
    
    def frac_var(self,n):
        total_var = np.sum(self.eigenvalues)
        f_var = self.eigenvalues[n-1]/total_var
        return f_var
    
    def homogeneous_map_left(self,n):
        t_x = self.expansion_coef_left(n)
        sigma_tx = np.std(t_x)
        tx_sigma = t_x/sigma_tx
        homogeneous_map_X = (self.X).dot(tx_sigma.T) / tx_sigma.dot(tx_sigma.T) 
        return pattern_maps(self.dato1,homogeneous_map_X)

    def homogeneous_map_right(self,n):
        t_y = self.expansion_coef_right(n)
        sigma_ty = np.std(t_y)
        ty_sigma = t_y/sigma_ty
        homogeneous_map_Y = (self.Y).dot(ty_sigma.T) / ty_sigma.dot(ty_sigma.T) 
        return pattern_maps(self.dato2,homogeneous_map_Y)
    
    def heterogeneous_map_left(self,n):
        t_y = self.expansion_coef_right(n)
        t_x = self.expansion_coef_left(n)
        sigma_ty = np.std(t_y)
        ty_sigma = t_y/sigma_ty
        heterogeneous_map_X = (self.X).dot(ty_sigma.T) / ty_sigma.dot(ty_sigma.T) * (1/np.corrcoef(t_x, t_y)[0,1])
        return pattern_maps(self.dato1,heterogeneous_map_X)
    
    def heterogeneous_map_right(self,n):
        t_x = self.expansion_coef_left(n)
        t_y = self.expansion_coef_right(n)
        sigma_tx = np.std(t_x)
        tx_sigma = t_x/sigma_tx
        heterogeneous_map_Y = (self.Y).dot(tx_sigma.T) / tx_sigma.dot(tx_sigma.T) * (1/np.corrcoef(t_x, t_y)[0,1])
        return pattern_maps(self.dato2,heterogeneous_map_Y)    
    
    def plot_homogeneous(self,n,proj):
        mapa1 = self.homogeneous_map_left(n)
        mapa2 = self.homogeneous_map_right(n)
        t_x = self.expansion_coef_left(n)
        t_y = self.expansion_coef_right(n)
        frac_var = self.frac_var(n) * 100
        if proj == 'stereo':
            fig = self.plot_stereo_map(mapa1,mapa2,t_x,t_y,'Homogeneous maps '+str(np.round(frac_var,3))+'%')
        else:
            fig = self.plot_map(mapa1,mapa2,t_x,t_y,'Homogeneous maps '+str(np.round(frac_var,3))+'%')
        return fig
    
    def plot_heterogeneous(self,n,proj):
        mapa1 = self.heterogeneous_map_left(n)
        mapa2 = self.heterogeneous_map_right(n)
        t_x = self.expansion_coef_left(n)
        t_y = self.expansion_coef_right(n)
        frac_var = self.frac_var(n) * 100
        if proj == 'stereo':
            fig = self.plot_stereo_map(mapa1,mapa2,t_x,t_y,'Heterogeneous maps '+str(np.round(frac_var,3))+'%')
        else:
            fig = self.plot_map(mapa1,mapa2,t_x,t_y,'Heterogeneous maps '+str(np.round(frac_var,3))+'%')
        return fig
    
    def plot_patterns(self,n,proj):
        mapa1, mapa2 = self.eof_pattern(n)
        t_x = self.expansion_coef_left(n)
        t_y = self.expansion_coef_right(n)
        frac_var = self.frac_var(n) * 100
        if proj == 'stereo':
            fig = self.plot_stereo_map(mapa1,mapa2,t_x,t_y,'pattern maps '+str(np.round(frac_var,3))+'%')
        else:
            fig = self.plot_map(mapa1,mapa2,t_x,t_y,'pattern maps '+str(np.round(frac_var,3))+'%')
        return fig
    
    def plot_stereo_map(self,mapa1,mapa2,t_x,t_y,title):
        fig = plt.figure(figsize=(20,10))
        gs = fig.add_gridspec(3, 4)
        ax1 = fig.add_subplot(gs[0:2, :2], projection=ccrs.SouthPolarStereo(central_longitude=300))
        ax2 = fig.add_subplot(gs[0:2, 2:], projection=ccrs.Robinson(180))
        clevels1 = np.arange(-1,1.1,0.1)
        clevels2 = np.arange(-1,1.1,0.1)
        ax1, im1 = plot_stereo(ax1,self.dato1.lat, self.dato1.lon, mapa1/np.max(mapa1),clevels1,cmap='PuOr')
        cbar = plt.colorbar(im1, orientation='vertical')
        im2 = plot_fig(ax2,self.dato2.lat, self.dato2.lon, mapa2/np.max(mapa2),ccrs.PlateCarree(),clevels2,cmap='RdBu_r')
        cbar = plt.colorbar(im2, orientation='vertical')
        ax3 = fig.add_subplot(gs[2, :2])
        ax4 = fig.add_subplot(gs[2, 2:])
        ax3.plot(t_x) ; ax3.set_ylabel('expansion coeficient left') ; ax3.set_xlabel('Years') 
        ax4.plot(t_y) ; ax4.set_ylabel('expansion coeficient right') ; ax4.set_xlabel('Years') 
        ax3.set_xticks(np.arange(0,len(t_x),3*10));ax3.set_xticklabels(np.arange(self.year1,self.year2+1,10))# ax3.set_xticklabels(np.arange(self.year1,self.year2,10))
        ax4.set_xticks(np.arange(0,len(t_x),3*10));ax4.set_xticklabels(np.arange(self.year1,self.year2+1,10))# ax4.set_xticklabels(np.arange(self.year1,self.year2,10))
        fig.suptitle(title)
           
        return fig
    
    def plot_map(self,mapa1,mapa2,t_x,t_y,title):
        fig = plt.figure(figsize=(20,10))
        gs = fig.add_gridspec(3, 4)
        ax1 = fig.add_subplot(gs[0:2, :2], projection=ccrs.PlateCarree(180))
        ax2 = fig.add_subplot(gs[0:2, 2:], projection=ccrs.Robinson(180))
        clevels1 = np.arange(-1,1.1,0.1)
        clevels2 = np.arange(-1,1.1,0.1)
        im1 = plot_fig(ax1,self.dato1.lat, self.dato1.lon, mapa1/np.max(mapa1),ccrs.PlateCarree(),clevels1,cmap='PuOr')
        cbar = plt.colorbar(im1, orientation='vertical')
        im2 = plot_fig(ax2,self.dato2.lat, self.dato2.lon, mapa2/np.max(mapa2),ccrs.PlateCarree(),clevels2,cmap='RdBu_r')
        cbar = plt.colorbar(im2, orientation='vertical')
        ax3 = fig.add_subplot(gs[2, :2])
        ax4 = fig.add_subplot(gs[2, 2:])
        ax3.plot(t_x) ; ax3.set_ylabel('expansion coeficient left') ; ax3.set_xlabel('Years') 
        ax4.plot(t_y) ; ax4.set_ylabel('expansion coeficient right') ; ax4.set_xlabel('Years') 
        ax3.set_xticks(np.arange(0,len(t_x),3*10));ax3.set_xticklabels(np.arange(self.year1,self.year2+1,10))# ax3.set_xticklabels(np.arange(self.year1,self.year2,10))
        ax4.set_xticks(np.arange(0,len(t_x),3*10));ax4.set_xticklabels(np.arange(self.year1,self.year2+1,10))# ax4.set_xticklabels(np.arange(self.year1,self.year2,10))
        fig.suptitle(title)
           
        return fig
    
    def summary(self,n):
        dic = {}
        dic['coef_left'] = {}
        dic['coef_right'] = {}
        dic['homogeneous_maps'] = {}
        dic['homogeneous_maps']['right'] = {}
        dic['homogeneous_maps']['left'] = {}
        dic['heterogeneous_maps'] = {}
        dic['heterogeneous_maps']['right'] = {}
        dic['heterogeneous_maps']['left'] = {}
        for i in range(n):
            dic['coef_left'][i] = self.expansion_coef_left(i)
            dic['coef_right'][i] = self.expansion_coef_right(i)
            dic['homogeneous_maps']['left'][i] = self.homogeneous_map_left(i)
            dic['homogeneous_maps']['right'][i] = self.homogeneous_map_right(i)
            dic['heterogeneous_maps']['left'][i] = self.heterogeneous_map_left(i)
            dic['heterogeneous_maps']['right'][i] = self.heterogeneous_map_right(i) 
            
        return dic
            


