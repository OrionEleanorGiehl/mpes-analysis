# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:15:40 2025

@author: lloyd
""" 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
import xarray as xr
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from lmfit import Model
from sigfig import round
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv

#%% Useful Functions and Definitions for Manipulating Data

# Partition Data into + and - Delays
def get_data_chunks(I, neg_times, t0, ax_delay_offset):
    if I.ndim > 3:
        tnf1 = (np.abs(ax_delay_offset - neg_times[0])).argmin()
        tnf2 = (np.abs(ax_delay_offset - neg_times[1])).argmin()

        I_neg = I[:,:,:,tnf1:tnf2+1] #Sum over delay/polarization/theta...
        neg_length = I_neg.shape[3]
        I_neg = I_neg
        I_neg_sum = I_neg.sum(axis=(3))/neg_length
    
        I_pos = I[:,:,:,t0+1:]
        pos_length = I_pos.shape[3]
        I_pos = I_pos #Sum over delay/polarization/theta...
        I_pos_sum = I_pos.sum(axis=(3))/pos_length
    
        I_sum = I[:,:,:,:].sum(axis=(3))
        
        return I_sum, I_pos, I_pos_sum, I_neg, I_neg_sum

    else:
        I_neg = I[:,:,:] #Sum over delay/polarization/theta...
        I_pos = I[:,:,:]
        I_sum = I

# Function for Creating MM Constant Energy kx, ky slice 
def get_momentum_map(I_res, E, E_int, delay=None, delay_int=None, **kwargs):
    subtract_neg = kwargs.get('subtract_neg', False)
    neg_delays = kwargs.get('neg_delays', (-250,-100))
    # Integrate over energy window
    I_E = I_res.loc[{"E":slice(E - E_int / 2, E + E_int / 2)}].mean(dim="E")
    if subtract_neg:
        I_E = I_E - I_E.loc[{"delay":slice(neg_delays[0],neg_delays[1])}].mean(dim='delay')

    if "delay" in I_res.dims:
        if delay is not None and delay_int is not None:
            # Integrate over a delay window
            frame = I_E.loc[{"delay":slice(delay - delay_int / 2, delay + delay_int / 2)}].mean(dim="delay").T
        else:
            # No delay specified: average over entire delay axis
            frame = I_E.mean(dim="delay").T
        
        frame = frame.assign_attrs(delay=delay, delay_int=delay_int)
    else:
        frame = I_E.T

    frame = frame.assign_attrs(E=E, E_int=E_int)
    return frame

def get_kx_E_frame(I_res, ky, ky_int, delay, delay_int):
    
    I_ky = I_res.loc[{"ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim="ky")

    if "delay" in I_res.dims:
        if delay is not None and delay_int is not None:
            # Integrate over a delay window
            frame = I_ky.loc[{"delay":slice(delay - delay_int / 2, delay + delay_int / 2)}].mean(dim="delay")
        else:
            # No delay specified: average over entire delay axis
            frame = I_ky.mean(dim="delay")
    else:
        frame = I_ky
                    
    return frame

def get_ky_E_frame(I_res, kx, kx_int, delay, delay_int):

    I_kx = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2)}].mean(dim="kx")

    if "delay" in I_res.dims:
        if delay is not None and delay_int is not None:
            # Integrate over a delay window
            frame = I_kx.loc[{"delay":slice(delay - delay_int / 2, delay + delay_int / 2)}].mean(dim="delay")
        else:
            # No delay specified: average over entire delay axis
            frame = I_kx.mean(dim="delay")
    else:
        frame = I_kx
                    
    return frame

def get_waterfall(I_res, kx, kx_int, ky=None, ky_int=None):
    
    #cmap = kwargs.get("cmap", "viridis")

    if "angle" in I_res.dims:
        angle = kx
        angle_int = kx_int
        frame = I_res.loc[{"angle":slice(angle-angle_int/2, angle+angle_int/2)}].mean(dim=("angle"))
    else:

        frame = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx","ky"))

    return frame

def get_k_cut(I, k_start, k_end, delay, delay_int, n, w):
    """
    Extract an E vs k slice along an arbitrary line in kx-ky space.

    Parameters:
    - I: xarray.DataArray with dims ('kx', 'ky', 'energy')
    - k_start: (kx0, ky0) tuple — start point in k-space
    - k_end: (kx1, ky1) tuple — end point in k-space
    - num_k: number of points along the k-space cut

    Returns:
    - I_cut: 2D array of shape (len(E), num_k)
    - k_vals: 1D array of k-distance along the cut
    - E_vals: 1D array of energies
    """
    num_k=n
    
    if "delay" in I.dims:
        if delay is not None and delay_int is not None:
            # Integrate over a delay window
            I = I.loc[{"delay":slice(delay - delay_int / 2, delay + delay_int / 2)}].mean(dim="delay")
        else:
            # No delay specified: average over entire delay axis
            I = I.mean(dim="delay")
    else:
        I = I
                    
    # Coordinate arrays
    kx_vals = I.kx.values
    ky_vals = I.ky.values
    E_vals = I.E.values

    # Create k-space cut line
    # Define the main k-line
    k_start, k_end = np.array(k_start), np.array(k_end)
    k_line = np.linspace(k_start, k_end, num_k)
    kx_line, ky_line = k_line[:, 0], k_line[:, 1]

    # Unit vectors
    d_vec = (k_end - k_start)
    d_vec /= np.linalg.norm(d_vec)
    n_vec = np.array([-d_vec[1], d_vec[0]])  # perpendicular unit vector

    # Interpolator
    interp = RegularGridInterpolator(
        (kx_vals, ky_vals, E_vals),
        I.transpose('kx', 'ky', 'E').values,
        bounds_error=False,
        fill_value=np.nan
    )

    # Precompute width offsets
    if w > 0:
        w_offsets = np.linspace(-w/2, w/2, 20)
    else:
        w_offsets = np.array([0.0])

    # Prepare all sampling points
    I_cut = np.zeros((len(E_vals), num_k))
    for i, (kx_i, ky_i) in enumerate(zip(kx_line, ky_line)):
        # Offset coordinates across the perpendicular direction
        kx_offsets = kx_i + n_vec[0] * w_offsets
        ky_offsets = ky_i + n_vec[1] * w_offsets

        # Build grid for each offset and energy
        kx_grid = np.repeat(kx_offsets[:, None], len(E_vals), axis=1)
        ky_grid = np.repeat(ky_offsets[:, None], len(E_vals), axis=1)
        E_grid = np.tile(E_vals[None, :], (len(w_offsets), 1))

        pts = np.column_stack([kx_grid.ravel(), ky_grid.ravel(), E_grid.ravel()])
        vals = interp(pts).reshape(len(w_offsets), len(E_vals))
        I_cut[:, i] = np.nanmean(vals, axis=0)

    # For each point along the k-line, extract the I(E) spectrum
    #I_cut = []
    #for kx_i, ky_i in zip(kx_line, ky_line):
    #    pts = np.column_stack([np.full_like(E_vals, kx_i),
    #                           np.full_like(E_vals, ky_i),
    #                           E_vals])
    #    spectrum = interp(pts)
    #    I_cut.append(spectrum)
    #I_cut = np.array(I_cut).T  # shape: (energy, k_index)

    # Compute 1D distance along cut
    dk = np.linalg.norm(k_end - k_start)
    k_vals = np.linspace(0, dk, num_k)

    k_frame = xr.DataArray(
        I_cut,
        dims=("E", "k"),
        coords={"E": E_vals, "k": k_vals},
        name="arb. k_cut"
    )

    return k_frame


def get_time_trace(I_res, E, E_int, k, k_int, norm_trace = False, **kwargs):
    # At the top of your function
    #if isinstance(k, (int, float)):
    #    k = (k,)
    #if isinstance(k_int, (int, float)):
     #   k_int = (k_int,)

    subtract_neg = kwargs.get("subtract_neg", False)
    neg_delays = kwargs.get("neg_delays", [-200, -100])

    d1, d2 = neg_delays[0], neg_delays[1]
    
    if "angle" in I_res.dims:
        #(angle,) = k
        #(angle_int,) = k_int
        angle, angle_int = k, k_int
        trace = I_res.loc[{"E":slice(E-E_int/2, E+E_int/2), "angle":slice(angle-angle_int/2, angle+angle_int/2)}].mean(dim=("angle", "E"))
    
    elif "kx" in I_res.dims and "ky" in I_res.dims:
        (kx, ky) = k
        (kx_int, ky_int) = k_int
        trace = I_res.loc[{"E":slice(E-E_int/2, E+E_int/2), "kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx", "ky", "E"))
    
    else:
        raise ValueError("Data must contain either ('angle') or ('kx', 'ky') dimensions.")

    if subtract_neg is True : 
        trace = trace - np.mean(trace.loc[{"delay":slice(d1,d2)}])
    
    if norm_trace is True : 
        trace = trace/np.max(trace)
    elif norm_trace is False:
        trace = trace
    else:
        trace = trace/norm_trace

    return trace

def get_edc(I_res, kx, ky, kx_int, ky_int, delay=0, delay_int=4000):
        
    if I_res.ndim > 3:    
        edc = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2), "delay":slice(delay-delay_int/2,delay+delay_int/2)}].mean(dim=("kx", "ky", "delay"))
    else:
        edc = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx", "ky"))

    return edc

def enhance_features(I_res, Ein, factor, norm):
    
    I1 = I_res.loc[{"E":slice(-6,Ein)}]
    I2 = I_res.loc[{"E":slice(Ein,6.5)}]

    if norm is True:
        I1 = I1/np.max(I1)
        I2 = I2/np.max(I2)
    else:
        I1 = I1/factor[0]
        I2 = I2/factor[1]
        
    I3 = xr.concat([I1, I2], dim = "E")
    
    return I3

def find_E0(edc, energy_window, p0, fig, ax):
    
    def gaussian(x, amp_1, mean_1, stddev_1, offset):
        
        g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)+offset
        
        return g1
    
    #plt.legend(frameon = False)
    ax[1].set_xlim([-1, 1]) 
    #ax[1].set_ylim([0, 1.1])
    ax[1].set_xlabel('E - E$_{{VBM}}$, eV')
    ax[1].set_ylabel('Norm. Int.')
    #ax[1].axvline(0, color = 'black', linestyle = 'dashed', linewidth = 0.5)
    #ax[1].set_yscale('log')
    #plt.ax[1].gca().set_aspect(2)
    
    ##### VBM #########
    #e1 = -.15
    #e2 = 0.6
    #    p0 = [1, .02, 0.17, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, -.155, 0.0, 0), (1.5, 0.75, 1.5, .5))

    e1, e2 = energy_window[0], energy_window[1]    
    try:
        popt, pcov = curve_fit(gaussian, edc.loc[{"E":slice(e1,e2)}].E.values, edc.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
    except ValueError:
        popt = [0, 0, 0, 0]
        pcov = [0, 0, 0, 0]
        print('oops!')
        
    perr = np.sqrt(np.diag(pcov))
            
    vb_fit = gaussian(edc.E, *popt)
    ax[1].plot(edc.E, edc, color = 'black', label = 'Data')
    ax[1].plot(edc.E, vb_fit, linestyle = 'dashed', color = 'red', label = 'Fit')
    ax[1].legend(frameon=False, loc = 'upper left', fontsize = 11)
    print(fr'E_VBM = {popt[1]:.3f} +- {perr[1]:.3f} eV')
    
def find_t0(trace_ex, delay_limits, fig=None, ax=None, **kwargs):
    
    norm = kwargs.get("norm", False)

    def rise_erf(t, t0, tau):
        r = 0.5 * (1 + erf((t - t0) / (tau)))
        return r
            
    p0 = [0.050, 0.045]
    #delay_limits = [-200,60]

    delay_axis = trace_ex.loc[{"delay":slice(delay_limits[0],delay_limits[1])}].delay.values
    delay_trace = trace_ex.loc[{"delay":slice(delay_limits[0],delay_limits[1])}].values
    
    if norm is True:
        delay_trace = delay_trace/np.max(delay_trace)

    popt, pcov = curve_fit(rise_erf, delay_axis, delay_trace, p0, method="lm")
    
    perr = np.sqrt(np.diag(pcov))
    
    rise_fit = rise_erf(np.linspace(delay_limits[0],delay_limits[1], 50), *popt)
    
    if fig is not None:
        
        ax[1].plot(trace_ex.delay, trace_ex, 'ko',label='Data')
        ax[1].plot(np.linspace(delay_limits[0],delay_limits[1],50), rise_fit, 'red',label='Fit')
        #ax[1].plot(I_res.delay, rise, 'red')
        ax[1].set_xlabel(r'Delay $\Delta t$ [ps]')
        ax[1].set_ylabel('Norm. Int.')
        ax[1].axvline(0, color = 'grey', linestyle = 'dashed')
        
        ax[1].set_xlim([-0.4, 0.4]) 
        ax[1].set_ylim(-.1,1.05)
        #ax[1].axvline(30)
        ax[1].legend(frameon=False)

    print(fr't0 = {popt[0]:.3f} +/- {perr[0]:.3f} ps')
    print(fr'width = {popt[1]:.3f} +/- {perr[1]:.3f} ps')
    
    return popt, perr, rise_fit


def t0_alt(I_res, Labels, fig=None, ax=None, **kwargs):
    '''
    Determines t0 by fitting an Error Function to the VB Dynamics
    
    params:
    - I_res: List of Normalized Spectra
    - Labels: List of Labels
    - fig: Figure to which the Result is plotted. If not specified, no plot is created
    - ax: Ax to which the Result is plotted. If not specified, no plot is created

    Optional kwargs:
    - kx: x-Coordinate for Center of Momentum-Integration (default: 0)
    - ky: y-Coordinate for Center of Momentum-Integration (default: 0)
    - kx_int: x-Width of Momentum-Integration (default: 4)
    - ky_int: y-Width of Momentum-Integration (default: 4)
    - delay_limits: List of Tuples of Delay Limits over which the Fits will be evaluated (default: [(-0.2, 0.15)])
    - neg_delays: List of Tuples of Negative Delays which will be subtracted from the Time Traces (default [(-0.3, -0.1)])
    - E_limits: Energy Integration range over which the Time Traces will be evaluated. A Window near the VB works best. (default: (0, 0.2))
    - xlims: Tuple of Limits for the x-Axis of the plot (default: (-0.3, 0.5))
    - fontsize: Fontsize. Legend and Ticklabels are 3 smaller than main Fontsize. (default: 15)
    '''
    kx = kwargs.get('kx', 0)
    ky = kwargs.get('ky', 0)
    kx_int = kwargs.get('kx_int', 4)
    ky_int = kwargs.get('ky_int', 4)
    delay_limits = kwargs.get('delay_limits', [(-200, 150)])
    neg_delays = kwargs.get('neg_delays', [(-300, -100)])
    E_limits = kwargs.get('E_limits', (0, 0.2))
    xlims = kwargs.get('xlims', (-300, 500))
    fontsize = kwargs.get('fontsize', 15)

    E_center = (E_limits[1] - E_limits[0])/2
    E_int = (E_limits[1] + E_limits[0])/2

    N = len(I_res)
    colormap = mpl.colormaps['viridis']
    colors = kwargs.get('colors', colormap(np.linspace(0.2,1, N)))
    
    delay_axs = []
    I_clipped = []
    for i in range(N):
        I_clipped.append(I_res[i].loc[{'E':slice(E_limits[0], E_limits[1])}])
        delay_axs.append(I_clipped[i].delay.values)
        delay_axs[i] = np.array(delay_axs[i])

    # Get TimeTraces in defined Delay Range
    TimeTraces = []
    for i in range(N):
        TimeTraces.append(get_time_trace(I_clipped[i], E_center, E_int, norm_trace=True, subtract_neg=True, neg_delays=neg_delays[i], k=(kx, ky), k_int=(kx_int, ky_int)))
    
    def rise_erf(t, t0, tau):
        r = 0.5 * (1 + erf((t - t0) / (tau)))
        return r
    
    initial_params = [0, 50] #t0, tau

    # Clip TimeTrace to delay Range
    TimeTraces_clipped = []
    delay_axs_clipped = []
    for i in range(N):
        temp = TimeTraces[i].loc[{'delay':slice(delay_limits[i][0], delay_limits[i][1])}]
        TimeTraces_clipped.append(temp.values)
        delay_axs_clipped.append(temp.delay.values)

    # Prepare Lists for optimal parameters, covariance matrix & uncertainties of optimal parameters
    optimal_params = [i for i in range(N)]
    covar_matrix = [i for i in range(N)]
    uncertainties = [i for i in range(N)]

    # Determine optimal parameters through curve fit annd calculate uncertainties
    for i in range(N):
        optimal_params[i], covar_matrix[i] = curve_fit(rise_erf, delay_axs_clipped[i], TimeTraces_clipped[i], initial_params)
        uncertainties[i] = np.sqrt(np.diag(covar_matrix[i]))
    
    # Read out t0
    t0 = [optimal_params[i][0] for i in range(N)]
    t0_strings = []
    for i in range(N):
        t0_strings.append(round(str(t0[i]), uncertainties[i][0], cutoff=19, sep='external_brackets'))
        #print(fr'{Labels[i]:25}: t0 = ({t0[i]:6.3f} +- {uncertainties[i][0]:.3f}) ps')
        print(fr'{Labels[i]:25}: t0 = {t0_strings[i]} fs')
        t0_strings[i] = fr'$t_0$ = {t0_strings[i]} fs'

    if fig is not None:
        t_plot = []
        t_tail0 = []
        t_tail1 = []
        for i in range(N):
            t_plot.append(np.linspace(delay_limits[i][0], delay_limits[i][1], num=1000))
            t_tail0.append(np.linspace(delay_axs[0].min(), delay_limits[i][0], num=500))
            t_tail1.append(np.linspace(delay_limits[i][1], delay_axs[0].max(), num=500))

        fit_curve = [rise_erf(t_plot[i], *optimal_params[i]) for i in range(N)]
        fit_tail0 = [rise_erf(t_tail0[i], *optimal_params[i]) for i in range(N)]
        fit_tail1 = [rise_erf(t_tail1[i], *optimal_params[i]) for i in range(N)]

        
        markerstyles = ['o', 's', 'D', '^', 'H', 'P', '8']

        for i in range(N):
            #ax.vlines(t0[i], ymin=-0.1, ymax=1.05, color=colors[i], alpha=0.5,
            #          path_effects=[pe.Stroke(linewidth=3, foreground='black', alpha=0.5), pe.Stroke(foreground='white', alpha=1), pe.Normal()])
            #TimeTraces[i].plot.scatter(label=Labels[i], marker=markerstyles[i], color=colors[i], alpha=0.5, edgecolors='black')
            TimeTraces[i].plot(label=Labels[i], color=colors[i], alpha=0.7, lw=3,
                    path_effects=[pe.Stroke(linewidth=6, foreground='black', alpha=0.6), pe.Stroke(foreground='white', alpha=1), pe.Normal()])
            #ax.plot(t_tail0[i], fit_tail0[i], color=colors[i], alpha=0.5, ls='dotted')
            #ax.plot(t_tail1[i], fit_tail1[i], color=colors[i], alpha=0.5, ls='dotted')
            #ax.plot(t_plot[i], fit_curve[i], color=colors[i], alpha=0.9, lw=2.5, label=t0_strings[i],
            #        path_effects=[pe.Stroke(linewidth=5, foreground='black', alpha=0.5), pe.Stroke(foreground='white', alpha=1), pe.Normal()])
            ax.plot(t_plot[i], fit_curve[i], color='black', lw=2.5, label=t0_strings[i], ls='solid')
            ax.vlines(t0[i], ymin=-0.1, ymax=1.05, color='black', ls='dashed', lw=2)

        
        ax.legend(fontsize=fontsize-1)
        #ax.axvline(delay_limits[0][0], color='grey', alpha=0.5)
        #ax.axvline(delay_limits[0][1], color='grey', alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(xlims[0], xlims[1])
        ax.tick_params(labelsize=fontsize-1)
        ax.set_xlabel(r'Delay $\Delta t$ [fs]', fontsize=fontsize)
        ax.set_ylabel('Normalized Intensity', fontsize=fontsize)

def VBMfromRisingEdge(I_res, Labels, fig=None, ax=None, **kwargs):
    '''
    Determines E_VBM by fitting a linear Function to the Rising Edge

    Parameters:
    - I_res: List of Normalized Spectra
    - Labels: List of Labels
    - fig: Figure to which the Result is plotted. If not specified, no plot is created
    - ax: Ax to which the Result is plotted. If not specified, no plot is created
    
    Optional kwargs:
    - kx: x-Coordinate for Center of Momentum-Integration (default: 0)
    - ky: y-Coordinate for Center of Momentum-Integration (default: 0)
    - kx_int: x-Width of Momentum-Integration (default: 4)
    - ky_int: y-Width of Momentum-Integration (default: 4)
    - E_limits: Tuple of Energy Limits over which the EDCs will be evaluated. Smaller Interval can significantly improve Computation Time. (default: (-6, 1))
    - fit_limits: Tuple of Energy Limits in which E_VBM is expected to be found (default: (-0.5, 0.5))
    - show_fit_lims: Bool deciding if the Fit Limits are plotted with grey lines as a visual aid (default: True)
    - fontsize: Fontsize. Legend and Ticklabels are 3 smaller than main Fontsize. (default: 15)
    '''
    kx = kwargs.get('kx', 0)
    ky = kwargs.get('ky', 0)
    kx_int = kwargs.get('kx_int', 4)
    ky_int = kwargs.get('ky_int', 4)
    E_limits = kwargs.get('E_limits', (-6,1))
    fit_limits = kwargs.get('fit_limits', (-0.5, 0.5))
    show_fit_lims = kwargs.get('show_fit_lims', True)
    fontsize = kwargs.get('fontsize', 15)
    
    colormap = mpl.colormaps['viridis']
    N = len(I_res)
    colors = kwargs.get('colors', colormap(np.linspace(0.2,1, N)))

    E_axs = [] # Make Energy Axes for plotting
    I_clipped = []
    for i in range(N):
        I_clipped.append(I_res[i].loc[{"E":slice(E_limits[0], E_limits[1])}]) # Spectra get clipped to E_limits to save time on EDC Computations
        E_axs.append(I_clipped[i].E.values)
        E_axs[i] = np.array(E_axs[i])

    # Get EDCs in defined Energy Range
    EDCs = []
    print(fr'Computing EDCs... (0/{N})', end='\r')
    for i in range(N):
        EDCs.append(get_edc(I_clipped[i], kx, ky, kx_int, ky_int))
        print(fr'Computing EDCs... ({i+1}/{N})', end='\r')
        EDCs[i] = EDCs[i]/np.max(EDCs[i])
        EDCs[i] = EDCs[i].loc[{"E":slice(E_limits[0], E_limits[1])}].values

    # Make Linear Fit Model to determine E0
    def Linear(E,E_VBM,a):
        return a*(E-E_VBM)

    initial_params = (0, 1) # E_VBM, a

    fit_indices = [] # List which will contain the indices over which the EDCs should be fitted
    E_fit = [] # Energy axes over which the fits will run
    EDCs_fit = [] # EDCs over which the fits will run
    # Find indices where Energy axis is between the fit limits and EDC is greater than 0.05, then write that section of Energy axes and EDCs into the lists
    for k in range(N):
        E_temp = E_axs[k]
        EDC_temp = EDCs[k]
        fit_indices.append([i for i in range(E_temp.shape[0]) if E_temp[i]>fit_limits[0] if E_temp[i]<fit_limits[1] if EDC_temp[i]>0.05])
        indices_temp = fit_indices[k]
        E_fit.append(E_temp[indices_temp[0]:indices_temp[-1]])
        EDCs_fit.append(EDC_temp[indices_temp[0]:indices_temp[-1]])

    # Prepare Lists for optimal parameters, covariance matrix & uncertainties of optimal parameters
    optimal_params = [i for i in range(N)]
    covar_matrix = [i for i in range(N)]
    uncertainties = [i for i in range(N)]

    # Determine optimal parameters through curve fit annd calculate uncertainties
    for i in range(N):
        optimal_params[i], covar_matrix[i] = curve_fit(Linear, E_fit[i], EDCs_fit[i], initial_params)
        uncertainties[i] = np.sqrt(np.diag(covar_matrix[i]))

    # Read out E_VBM
    E_VBM = [optimal_params[i][0] for i in range(N)]
    E_VBM_strings = []
    for i in range(N):
        E_VBM_strings.append(round(str(E_VBM[i]), uncertainties[i][0], cutoff=19, sep='external_brackets'))
        print(fr'{Labels[i]:25}: E_VBM = {E_VBM_strings[i]} eV')
        E_VBM_strings[i] = fr'$E_{{\mathrm{{VBM}}}}$ = {E_VBM_strings[i]} eV'

    if fig is not None:
        E_plot = [np.linspace(fit_limits[0], E_VBM[i]) for i in range(len(E_VBM))] # Best Fits should be plotted from lower bound until E_VBM

        fit_curve = [Linear(E_plot[i], *optimal_params[i]) for i in range(N)]

        
        FitLabels = []
        for i in range(N):
            FitLabels.append(fr'$E_{{\mathrm{{VBM}}}}$')
            ax.plot(E_axs[i], EDCs[i], color=colors[i], alpha=0.7, lw=3, label=Labels[i],
                    path_effects=[pe.Stroke(linewidth=6, foreground='black', alpha=0.6), pe.Stroke(foreground='white', alpha=1), pe.Normal()])
            #ax.plot(E_plot[i], fit_curve[i], color=colors[i], alpha=0.9, ls='dashed', lw=2.5, label=E_VBM_strings[i],
            #        path_effects=[pe.Stroke(linewidth=5, foreground='black', alpha=0.5), pe.Stroke(foreground='white', alpha=1), pe.Normal()])
            ax.plot(E_plot[i], fit_curve[i], color='black', ls='solid', lw=2.5, label=E_VBM_strings[i])
            #ax.vlines(optimal_params[i][0], ymin=-.05, ymax=1.05, color=colors[i], alpha=0.7, ls='solid',
            #          path_effects=[pe.Stroke(linewidth=3, foreground='black', alpha=0.5), pe.Stroke(foreground='white', alpha=1), pe.Normal()])
            ax.axvline(optimal_params[i][0], color='black', ls='dashed', lw=2)
    
        if show_fit_lims:
            ax.axvline(fit_limits[0], color='grey', alpha=0.5)
            ax.axvline(fit_limits[1], color='grey', alpha=0.5)
        ax.legend(fontsize=fontsize-1)
        ax.set_ylim(-.05,1.05)
        ax.set_xlim(E_limits[0], E_limits[1])
        ax.tick_params(labelsize=fontsize-1)
        ax.set_xlabel(fr'$E-E_{{\mathrm{{VBM}}}}$ [eV]', fontsize=fontsize)
        ax.set_ylabel(fr'Normalized Intensity', fontsize=fontsize)

# Useful Functions and Definitions for Plotting Data
def save_figure(fig, name, image_format):
    
    fig.savefig(name + '.'+ image_format, bbox_inches='tight', format=image_format)
    print('Figure Saved!')

def plot_edc(I, kx, ky, kx_int, ky_int, label=None, fig=None, ax=None, **kwargs):
    
    fontsize = kwargs.get("fontsize", 15)
    colormap = mpl.colormaps['viridis']
    color = kwargs.get('color', colormap(0.6))
    xlim = kwargs.get('xlim', (I.E.min(), I.E.max()))
    ylim = kwargs.get('ylim', (0,1.05))
    delay = kwargs.get('delay', 0)
    delay_int = kwargs.get('delay_int', 2000)
    
    if label is None:
        label =''

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(4,2), squeeze=False)
        ax = np.ravel(ax)
    else:
        ax = [ax]

    edc = get_edc(I, kx, ky, kx_int, ky_int, delay, delay_int)
    edc = edc/np.max(edc)
    
    edc.plot(ax=ax[0], color = color, alpha=0.7, lw=2, label=label,
                      path_effects=[pe.Stroke(linewidth=5, foreground='black', alpha=0.7), pe.Stroke(foreground='white', alpha=1), pe.Normal()])
    ax[0].set_xlim(xlim[0], xlim[1])
    ax[0].set_ylim(ylim[0], ylim[1])
    ax[0].set_xlabel(fr'$E-E_{{\mathrm{{VBM}}}}$ [eV]', fontsize=fontsize)
    ax[0].set_ylabel(fr'Normalized Intensity', fontsize=fontsize)

    fig.tight_layout()

def plot_momentum_maps(I, E, E_int, delays=None, delay_int=None, fig=None, ax=None, **kwargs):
    """
    Plot momentum maps at specified energies and delays with optional layout and styling.
    
    Parameters:
    - I: xarray dataset (e.g., I_diff or I_res).
    - E: list of energies.
    - E_int: total energy integration width (float).
    - delays: list of delays (same length as E). Ignored if no 'delay' in data.
    - delay_int: integration window for delays (float). Ignored if no 'delay' in data.
    
    Optional kwargs:
    - fig: matplotlib figure object (optional).
    - ax: array of axes (optional). If not provided, subplots are created.
    - cmap: colormap (default 'viridis').
    - scale: list [vmin, vmax] (default [0, 1]).
    - panel_labels: list of text labels (e.g., ['(a)', '(b)', ...]) (optional).
    - label_positions: tuple (x, y) in axes fraction coords for labels (default: (0.03, 0.9)).
    - fontsize: int for all text (default: 14).
    - figsize: tuple for fig size (only used if fig is created here).
    - nrows, ncols: layout for auto subplot creation (optional).
    - colorbar
    
    Returns:
    - fig, ax, im (image handle for colorbar)
    """
    E = np.atleast_1d(E)

    has_delay = "delay" in I.dims

    delays = np.atleast_1d(delays)
    
    if has_delay:
        delays = np.atleast_1d(delays)
        if len(delays) != len(E):
            if len(delays) < len(E):
                delays = np.resize(delays, len(E))
            elif len(delays) > len(E):
                E = np.resize(E, len(delays))
    else:  
        # Static data – ignore delays entirely
        delays = [None] * len(E)

    cmap = kwargs.get("cmap", "viridis")
    scale = kwargs.get("scale", [0, 1])
    panel_labels = kwargs.get("panel_labels", False)
    label_positions = kwargs.get("label_positions", (0.0, 1.1))
    fontsize = kwargs.get("fontsize", 14)
    figsize = kwargs.get("figsize", (8, 5))
    nrows = kwargs.get("nrows", 1)
    ncols = kwargs.get("ncols", int(np.ceil(len(E) / nrows)))
    colorbar = kwargs.get("colorbar", False)
    mask_radius = kwargs.get('mask_radius', None)
    subtract_neg = kwargs.get('subtract_neg', False)
    neg_delays = kwargs.get('neg_delays', (-250,-100))

    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        ax = np.ravel(ax)
    else:
        ax = [ax]
        
    for i in range(len(E)):
        frame = get_momentum_map(I, E[i], E_int, delays[i], delay_int, subtract_neg=subtract_neg, neg_delays=neg_delays)
        if mask_radius is not None:
            frame = frame.where((frame.kx**2 + frame.ky**2) < mask_radius**2, other=0)
        
        frame = frame / frame.max()

        im = frame.plot.imshow(
            ax=ax[i],
            vmin=scale[0],
            vmax=scale[1],
            cmap=cmap,
            add_colorbar=False
        )

        # Consistent formatting for all axes
        ax[i].set_aspect(1)
        ax[i].set_xlim(-2, 2)
        ax[i].set_ylim(-2, 2)
        ax[i].set_xticks(np.arange(-2, 2.2, 1))
        #for label in ax[i].xaxis.get_ticklabels()[1::2]:
        #    label.set_visible(False)

        ax[i].set_yticks(np.arange(-2, 2.1, 1))
        #for label in ax[i].yaxis.get_ticklabels()[1::2]:
        #    label.set_visible(False)

        ax[i].tick_params(labelsize=fontsize-1)
        ax[i].set_xlabel(r'$k_x$ [$\AA^{{-1}}]$', fontsize=fontsize)
        ax[i].set_ylabel(r'$k_y$ [$\AA^{{-1}}]$', fontsize=fontsize)
        ax[i].set_title(fr"$E-E_{{\mathrm{{VBM}}}}$ = ({E[i]:.2f} $\pm$ {E_int/2}) eV", fontsize=fontsize)

        # Optional panel label
        if panel_labels is True:
            labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
            ax[i].text(
                label_positions[0], label_positions[1],
                labels[i],
                transform=ax[i].transAxes,
                fontsize=fontsize,
                fontweight='regular'
            )
    
    if colorbar == True:
        # Add colorbar
        cbar_ax = fig.add_axes([1.02, 0.36, 0.025, 0.35])
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['min', 'max'])

    fig.tight_layout()

    return fig, ax, im

def SavePeak(opt_params_x, opt_params_y, unc_x, unc_y, x_orig, y_orig, x_ang, y_ang, csv_file):
    # Save Fit Params and Conditions of Gaussian Peaks along 2 directions
    data = [['axis', 'origin', 'k0', 'FWHM', 'Amplitude', 'constant'],
        ['x', x_orig[0], opt_params_x[0][0][0], opt_params_x[0][0][1], opt_params_x[0][0][2], opt_params_x[0][0][3]],
        [x_ang, x_orig[1], unc_x[0][0][0], unc_x[0][0][1], unc_x[0][0][2], unc_x[0][0][3]],
        ['y', y_orig[0], opt_params_y[0][0][0], opt_params_y[0][0][1], opt_params_y[0][0][2], opt_params_y[0][0][3]],
        [y_ang, y_orig[1], unc_y[0][0][0], unc_y[0][0][1], unc_y[0][0][2], unc_y[0][0][3]]]
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def LoadPeak(file):
    # Load Fit Params and Conditions of Gaussian Peaks along 2 directions
    array = np.genfromtxt(file, delimiter=',')
    x_origin = (array[1][1], array[2][1])
    x_rotation = array[2][0]
    (x_k0, x_FWHM, x_A) = (array[1][2], array[1][3], array[1][4]) # (k0, FWHM, Amplitude)
    x_uncertainties = (array[2][2], array[2][3], array[2][4])
    y_origin = (array[3][1], array[4][1])
    y_rotation = array[4][0]
    (y_k0, y_FWHM, y_A) = (array[3][2], array[3][3], array[3][4]) # (k0, FWHM, Amplitude)
    y_uncertainties = (array[4][2], array[4][3], array[4][4])
    return x_k0, y_k0, x_FWHM, y_FWHM, (x_A+y_A)/2, x_origin, x_rotation, y_origin, y_rotation, x_uncertainties, y_uncertainties

def plot_kx_frame(I_res, ky, ky_int, delays = None, delay_int = None, fig=None, ax=None, **kwargs):
    """
    Plot time traces of momentum frame at specified kx for multiple energies.
    
    Parameters:
    - I_res: xarray dataset (e.g., I_diff or I_res).
    - E_list: list of energies for which to plot time traces.
    - E_int: energy integration width (float).
    - kx: the specific kx value to plot.
    - kx_int: kx integration window (float).
    - delays: list of delay values.
    - delay_int: delay integration window (float).
    
    Optional kwargs:
    - fig: matplotlib figure object (optional).
    - ax: axis (optional). If not provided, a new axis is created.
    - cmap: colormap (default 'viridis').
    - norm_trace: whether to normalize the trace (default: True).
    - subtract_neg: whether to subtract the negative delays (default: True).
    - neg_delays: the range of negative delays to subtract (default: (-3, 0)).
    - fontsize: int for all text (default: 14).
    - nrows, ncols: layout for auto subplot creation (optional).
    - colorbar
    """
    has_delay = "delay" in I_res.dims

    delays = np.atleast_1d(delays)
    
    if has_delay:
        delays = np.atleast_1d(delays)
    else:
        # Static data – ignore delays entirely
        delays = [None]

    nrows = kwargs.get("nrows", 1)
    ncols = kwargs.get("ncols", int(np.ceil(len(delays) / nrows)))
    figsize = kwargs.get("figsize", (8, 5))
    fontsize = kwargs.get("fontsize", 14)
    cmap = kwargs.get("cmap", "viridis")
    scale = kwargs.get("scale", [0, 1])
    energy_limits=kwargs.get("energy_limits", (1,3))
    E_enhance = kwargs.get("E_enhance", None)

    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        ax = np.ravel(ax)
    else:
        ax = [ax]
    
    # Loop over the energy list to plot time traces at each energy
    for i, delay in enumerate(delays):
        # Get the frame for the given energy, kx, and delay
        kx_frame = get_kx_E_frame(I_res, ky, ky_int, delay, delay_int)
        if E_enhance is not None:    
            kx_frame = enhance_features(kx_frame, E_enhance, factor = 0, norm = True)
            ax[i].axhline(E_enhance, linestyle = 'dashed', color = 'white', linewidth = 1)

        im = kx_frame.T.plot.imshow(ax=ax[i], cmap=cmap, add_colorbar=False, vmin=scale[0], vmax=scale[1]) #kx, ky, t
        
        #ax[2].set_aspect(1)
        ax[i].set_xticks(np.arange(-2,2.2,1))
        for label in ax[i].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax[i].set_yticks(np.arange(energy_limits[0],energy_limits[1]+0.1,.25))
        for label in ax[i].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[i].set_xlabel(r'$k_x$ [$\AA^{{-1}}$]', fontsize = fontsize)
        ax[i].set_ylabel(r'$E - E_{\mathrm{VBM}}$ [eV]', fontsize = fontsize)
        ax[i].set_title(fr'$k_y$ = ({ky} $\pm$ {ky_int/2}) $\AA^{{-1}}$', fontsize = fontsize)
        ax[i].tick_params(axis='both', labelsize=fontsize-2)
        ax[i].set_xlim(-2,2)
        ax[i].set_ylim(energy_limits[0], energy_limits[1])
        #if has_delay and delays[0] is not None:
            #ax[i].text(-1.9, 2.7,  fr"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=14)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig, ax, im

def plot_ky_frame(I_res, kx, kx_int, delays=None, delay_int=None, fig=None, ax=None, **kwargs):
    """
    Plot time traces of momentum frame at specified kx for multiple energies.
    
    Parameters:
    - I_res: xarray dataset (e.g., I_diff or I_res).
    - E_list: list of energies for which to plot time traces.
    - E_int: energy integration width (float).
    - ky: the specific kx value to plot.
    - ky_int: kx integration window (float).
    - delays: list of delay values.
    - delay_int: delay integration window (float).
    
    Optional kwargs:
    - fig: matplotlib figure object (optional).
    - ax: axis (optional). If not provided, a new axis is created.
    - cmap: colormap (default 'viridis').
    - norm_trace: whether to normalize the trace (default: True).
    - subtract_neg: whether to subtract the negative delays (default: True).
    - neg_delays: the range of negative delays to subtract (default: (-3, 0)).
    - fontsize: int for all text (default: 14).
    - nrows, ncols: layout for auto subplot creation (optional).
    - colorbar
    """

    has_delay = "delay" in I_res.dims
    
    if has_delay:
        delays = np.atleast_1d(delays)
    else:
        # Static data – ignore delays entirely
        delays = [None]

    nrows = kwargs.get("nrows", 1)
    ncols = kwargs.get("ncols", int(np.ceil(len(delays) / nrows)))
    figsize = kwargs.get("figsize", (8, 5))
    fontsize = kwargs.get("fontsize", 14)
    cmap = kwargs.get("cmap", "viridis")
    scale = kwargs.get("scale", [0, 1])
    energy_limits=kwargs.get("energy_limits", (-4,2))
    E_enhance = kwargs.get("E_enhance", None)

    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        ax = np.ravel(ax)
    else:
        ax = [ax]
    
    # Loop over the energy list to plot time traces at each energy
    for i, delay in enumerate(delays):
        # Get the frame for the given energy, kx, and delay
        ky_frame = get_ky_E_frame(I_res, kx, kx_int, delay, delay_int)
        if E_enhance is not None:    
            ky_frame = enhance_features(ky_frame, E_enhance, factor = 0, norm = True)
            ax[i].axhline(E_enhance, linestyle = 'dashed', color = 'white', linewidth = 1)

        ky_frame.T.plot.imshow(ax=ax[i], cmap=cmap, add_colorbar=False, vmin=scale[0], vmax=scale[1]) #kx, ky, t
        
        #ax[2].set_aspect(1)
        ax[i].set_xticks(np.arange(-2,2.2,1))
        for label in ax[i].xaxis.get_ticklabels()[1::2]:
            label.set_visible(True)
        ax[i].set_yticks(np.arange(energy_limits[0],energy_limits[1]+0.1,.25))
        for label in ax[i].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[i].set_xlabel(r'$k_y$ [$\AA^{-1}$]', fontsize = fontsize)
        ax[i].set_ylabel(r'$E - E_{\mathrm{VBM}}$ [eV]', fontsize = fontsize)
        ax[i].set_title(fr'$k_x$ = ({kx} $\pm$ {kx_int/2}) $\AA^{{-1}}$', fontsize = fontsize)
        ax[i].tick_params(axis='both', labelsize=fontsize-2)
        ax[i].set_xlim(-2,2)
        ax[i].set_ylim(energy_limits[0], energy_limits[1])
        #ax[i].axhline(0.9, linestyle = 'dashed', color = 'black', linewidth = 1)
        #if has_delay and delays[0] is not None:
            #ax[i].text(-1.9, 2.7,  fr"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=14)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig, ax

def plot_k_cut(I_res, k_start, k_end, delays=None, delay_int=None, fig=None, ax=None, **kwargs):
        
    has_delay = "delay" in I_res.dims
    
    if has_delay:
        delays = np.atleast_1d(delays)
    else:
        # Static data – ignore delays entirely
        delays = [None]
    
    nrows = kwargs.get("nrows", 1)
    ncols = kwargs.get("ncols", int(np.ceil(len(delays) / nrows)))
    figsize = kwargs.get("figsize", (8, 5))
    fontsize = kwargs.get("fontsize", 14)
    cmap = kwargs.get("cmap", cmap_LTL)
    scale = kwargs.get("scale", [0, 1])
    energy_limits=kwargs.get("energy_limits", (-3,2.5))
    E_enhance = kwargs.get("E_enhance", None)
    ax2 = kwargs.get("ax2", None)
    n = kwargs.get("n", 200)
    w = kwargs.get("w", .2)

    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        ax = np.ravel(ax)
    else:
        ax = [ax]

    # Loop over the energy list to plot time traces at each energy
    for i, delay in enumerate(delays):
        # Get the frame for the given energy, kx, and delay
        k_frame = get_k_cut(I_res, k_start, k_end, delay, delay_int, n, w)
        k_frame = k_frame/np.max(k_frame)

        if E_enhance is not None:    
            k_frame = enhance_features(k_frame, E_enhance, factor = 0, norm = True)
            ax[i].axhline(E_enhance, linestyle = 'dashed', color = 'black', linewidth = 1)

        #im = ax[i].pcolormesh(k_vals, E_vals, k_cut, shading='auto', cmap=cmap_LTL)
        im = k_frame.plot.imshow(ax=ax[i], cmap=cmap, add_colorbar=False, vmin=scale[0], vmax=scale[1]) #kx, ky, t
        
        #ax[2].set_aspect(1)
        ax[i].set_xticks(np.arange(-2,3.5,.5))
        for label in ax[i].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax[i].set_yticks(np.arange(-4,4.1,0.5))
        for label in ax[i].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[i].set_xlabel(r'$k_{//}$, $\AA^{{-1}}$', fontsize = 18)
        ax[i].set_ylabel(r'$E - E_{{VBM}}, eV$', fontsize = 18)
        ax[i].set_title("E vs k slice", color = 'black', fontsize = 18)
        ax[i].tick_params(axis='both', labelsize=16)
        ax[i].set_xlim(0,k_frame.k.values.max())
        ax[i].set_ylim(energy_limits[0], energy_limits[1])
        
        if ax2 is not None:
            ax2.plot(k_start[0], k_start[1], color = 'purple', marker = 'o')
            ax2.plot(k_end[0], k_end[1], color = 'purple', marker = 'o')
            ax2.plot([k_start[0], k_end[0]], [k_start[1], k_end[1]], color = 'purple', linestyle = 'dashed')

        if has_delay and delays[0] is not None:
            ax[i].text(-1.9, 2.7,  fr"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=14)
        #ax[i].set_aspect(1)

    # Adjust layout
    fig.tight_layout()
    
    return fig, ax, im

def plot_time_traces(I_res, E, E_int, k, k_int, norm_trace=True, subtract_neg=True, neg_delays=(-500, -150), fig=None, ax=None, **kwargs):
    """
    Plot time traces at a specific energy and momentum coordinates with optional styling.
    
    Parameters:
    - I_res: xarray dataset (e.g., I_diff or I_res).
    - E: list of energies for the time trace plot.
    - kx, ky: momentum coordinates at which to extract the time trace.
    - kx_int, ky_int: momentum integration widths for the time trace.
    - E_int: energy integration width.
    - norm_trace: whether to normalize the trace (default: True).
    - subtract_neg: whether to subtract the mean of the negative delays (default: True).
    - neg_delays: range for background subtraction (default: (-200, -50)).
    - fig: matplotlib figure object (optional).
    - ax: axes object (optional).
    - panel_labels: list of panel labels (e.g., ['(a)', '(b)', ...]).
    - label_positions: position for panel labels (default: (0.03, 0.9)).
    - fontsize: font size for all text (default: 14).
    
    Returns:
    - fig, ax (figure and axis objects).
    """
    colormap = mpl.colormaps['viridis']
    fontsize = kwargs.get("fontsize", 14)
    
    legend = kwargs.get("legend", True)
    label = kwargs.get("label", None)

    #(kx, ky), (kx_int, ky_int) = k, k_int
    E = np.atleast_1d(E)
    colors = kwargs.get("colors", colormap(np.linspace(0.2,1, len(E))))
    #k = np.atleast_1d(k)
    #if len(E) > len(k):
     #   k = np.resize(k, len(E))
    #if len(E) < len(k):
    #    E = np.resize(E, len(k))        
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    for i in range(len(E)):
        trace = get_time_trace(I_res, E[i], E_int, k, k_int, norm_trace=norm_trace, subtract_neg=subtract_neg, neg_delays=neg_delays)
        ax.plot(trace.coords['delay'].values, trace.values, label=None, color = 'white', linewidth=2,
                path_effects=[pe.Stroke(linewidth=5, foreground='black', alpha=0.7), pe.Stroke(foreground='white', alpha=1), pe.Normal()], zorder=0)

    for i, E in enumerate(E):
        if label is None:
            label = f'$E-E_{{\mathrm{{VBM}}}}$ = {E:.2f} eV'

        trace = get_time_trace(I_res, E, E_int, k, k_int, norm_trace=norm_trace, subtract_neg=subtract_neg, neg_delays=neg_delays)
        ax.plot(trace.coords['delay'].values, trace.values, label=label, color = colors[i], linewidth=2.5, alpha=0.7, zorder=2)
    
    # Formatting
    ax.set_xlabel(fr'Delay $\Delta t$ [fs]', fontsize=fontsize)
    ax.set_ylabel('Intensity' , fontsize=fontsize)
    
    ax.xaxis.reset_ticks()
    ax.yaxis.reset_ticks()

    if trace.delay.values.max() < 1500:
        ax.set_xticks(np.arange(-1200, 1600, 100))
    else:
        ax.set_xticks(np.arange(-2000, 5000, 500))
    
    for label in ax.xaxis.get_ticklabels():
        label.set_visible(True)

    for label in ax.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    
    ax.set_yticks(np.arange(-0.5,1.25,0.25))
    for label in ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)

    if subtract_neg is True:
        ax.set_ylim(-0.1, 1.1)
    else:
        ax.set_ylim(0, 1.1)

    ax.tick_params(axis='both', labelsize=fontsize-1)
    ax.set_xlim(I_res.delay[1], I_res.delay[-1])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if legend is True:
        ax.legend(fontsize=fontsize-1, frameon=True)
    
    fig.tight_layout()

    return fig, ax

def plot_phoibos_frame(I_res, delay=None, delay_int=None, fig=None, ax=None, **kwargs):
    
    subtract_neg = kwargs.get("subtract_neg", False)
    #xlabel = kwargs.get("xlabel", 'Delay, ps')
    #ylabel = kwargs.get("ylabel", 'Intensity')
    fontsize = kwargs.get("fontsize", 14)
    figsize = kwargs.get("figsize", (8, 6))
    energy_limits=kwargs.get("energy_limits", (I_res.E.values[0],I_res.E.values[-1]))
    neg_delays = kwargs.get("neg_delays", [-500, -100])
    E_enhance = kwargs.get("E_enhance", None)

    if subtract_neg is True : 
        cmap = kwargs.get("cmap", cmap_LTL2)
        scale = kwargs.get("scale", [-1, 1])
    else:
        cmap = kwargs.get("cmap", cmap_LTL)
        scale = kwargs.get("scale", [0, 1])

    d1, d2 = neg_delays[0], neg_delays[1]
    
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if "delay" in I_res.dims:
        I_diff = I_res - I_res.loc[{"delay":slice(d1,d2)}].mean(dim='delay')

        if delay is not None:
            if subtract_neg is True: 
                frame = I_diff.loc[{"delay":slice(delay-delay_int/2,delay+delay_int/2)}].mean(dim='delay')

            else:
                frame = I_res.loc[{"delay":slice(delay-delay_int/2,delay+delay_int/2)}].mean(dim='delay')

        if delay is None:
            if subtract_neg is True: 
                frame = I_diff.mean(dim='delay')

            else:
                frame = I_res.mean(dim='delay')
    else:
        frame = I_res

    if E_enhance is not None:
        frame = enhance_features(frame, E_enhance, factor = 0, norm = True)
        ax.axhline(E_enhance, linestyle = 'dashed', color = 'black', linewidth = 1)
    else:
        frame = enhance_features(frame, energy_limits[0], factor = 0, norm = True)
    
    print(frame.shape)
    ph = frame.T.plot.imshow(ax = ax, vmin = scale[0], vmax = scale[1], cmap = cmap, add_colorbar=False)
   
    ax.set_xlabel('Angle', fontsize = fontsize)
    ax.set_ylabel(r'E - E$_{\mathrm{VBM}}$, eV', fontsize = fontsize)
    ax.set_yticks(np.arange(-5,3.5,0.5))
    ax.set_xlim(I_res.angle[1], I_res.angle[-1])
    ax.set_ylim(energy_limits[0], energy_limits[1])
    ax.set_title('Frame')
    ax.axhline(energy_limits[0], linestyle = 'dashed', color = 'black', linewidth = 1)
    
    for label in ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    #hor = I_res.delay[-1] - I_res.delay[1]
    #ver =  energy_limits[1] - energy_limits[0]
    #aspra = hor/ver 
    #ax[1].set_aspect(aspra)
    ax.set_aspect("auto")

    # Adjust layout to avoid overlap
    fig.tight_layout()   

def plot_waterfall(I_res, kx, kx_int, ky=None, ky_int=None, fig=None, ax=None, **kwargs):
    """
    Plot the waterfall of intensity across both kx and ky slices.

    Parameters:
    - I_res: xarray dataset with intensity data.
    - kx: kx value around which to extract the data (in 1/Å).
    - kx_int: integration window for kx (in 1/Å).
    - ky: ky value around which to extract the data (in 1/Å).
    - ky_int: integration window for ky (in 1/Å).

    Optional kwargs:
    - cmap: colormap for the waterfall plot (default 'viridis').
    - scale: [vmin, vmax] for normalization (default [0, 1]).
    - xlabel: label for the x-axis (default 'Delay, ps').
    - ylabel: label for the y-axis (default 'Intensity').
    - fontsize: font size for the labels (default: 14).
    - figsize: figure size (default (10, 6)).

    Returns:
    - fig, ax: figure and axis handles for the plot.
    """
    subtract_neg = kwargs.get("subtract_neg", False)

    if subtract_neg is True : 
    
        cmap = kwargs.get("cmap", cmap_LTL2)
        scale = kwargs.get("scale", [-1, 1])
    else:
        cmap = kwargs.get("cmap", cmap_LTL)
        scale = kwargs.get("scale", [0, 1])

    xlabel = kwargs.get("xlabel", 'Delay, ps')
    ylabel = kwargs.get("ylabel", 'Intensity')
    fontsize = kwargs.get("fontsize", 14)
    figsize = kwargs.get("figsize", (10, 6))
    energy_limits=kwargs.get("energy_limits", (1,3))
    neg_delays = kwargs.get("neg_delays", [-250, -120])
    E_enhance = kwargs.get("E_enhance", None)
    colorbar = kwargs.get('colorbar', False)

    d1, d2 = neg_delays[0], neg_delays[1]
    
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    waterfall = get_waterfall(I_res, kx, kx_int, ky, ky_int)

    if subtract_neg is True : 
        waterfall = waterfall - waterfall.loc[{"delay":slice(d1,d2)}].mean(dim='delay')

    if E_enhance is not None:
        waterfall = enhance_features(waterfall, E_enhance, factor = 0, norm = True)
        ax.axhline(E_enhance, linestyle = 'dashed', color = 'black', linewidth = 1.5)
    else:
        waterfall = enhance_features(waterfall, energy_limits[0], factor = 0, norm = True)
    
    wf = waterfall.plot.imshow(ax = ax, vmin = scale[0], vmax = scale[1], cmap = cmap, add_colorbar=colorbar)
    #waterfall.plot.imshow(ax = ax, cmap = cmap, add_colorbar=False)
   
    ax.set_xlabel(r'Delay $\Delta t$ [fs]', fontsize = fontsize) #used to say fs, but I'm pretty sure the scale in the data is picoseconds
    ax.set_ylabel(r'$E-E_{\mathrm{VBM}}$ [eV]', fontsize = fontsize)
    ax.set_yticks(np.arange(-1,3.5,0.25))
    ax.tick_params(axis='both', labelsize=fontsize-1)
    ax.set_xlim(I_res.delay[1], I_res.delay[-1])
    ax.set_ylim(energy_limits[0], energy_limits[1])
    #ax.set_title('$k$-Integrated')
    ax.axhline(energy_limits[0], linestyle = 'dashed', color = 'black', linewidth = 1)
    
    for label in ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    hor = I_res.delay[-1] - I_res.delay[1]
    ver =  energy_limits[1] - energy_limits[0]
    aspra = hor/ver 
    #ax[1].set_aspect(aspra)
    ax.set_aspect("auto")

    # Adjust layout to avoid overlap
    fig.tight_layout()

    return fig, ax, wf

def plot_mdcs(momentum_map, fig=None, ax=None, **kwargs):
    '''
    Plots a Momentum Map, then computes and plots two MDCs along arbitrary directions in the Momentum Map (Default kx- and ky-Axis).

    Returns: ax_mdc_kx, mdc_kx, ax_mdc_ky, mdc_ky

    params:
    - momentum_map: Momentum Map from which mdcs should be extracted

    Optional kwargs:
    - k_zero_x: [kx, ky]-Origin for the Coordinate System of kx-MDC (Default: [0,0])
    - k_zero_y: [kx, ky]-Origin for the Coordinate System of ky-MDC (Default: [0,0])
    - angle_xaxis: Rotation of kx-Axis for kx-MDC in Degrees (Default: 0)
    - angle_yaxis: Rotation of ky-Axis for ky-MDC in Degrees (Default: 0)
    - x_length: Length of kx-MDC-Cut (Default: 4)
    - y_length: Length of ky-MDC-Cut (Default: 4)
    - mdc_x_width: Integration width of kx-MDC (Default: 0.3)
    - mdc_y_width: Integration width of ky-MDC (Default: 0.3)
    '''
    k_zero_x = kwargs.get('k_zero_x', [0,0])
    k_zero_y = kwargs.get('k_zero_y', [0,0])
    x_length = kwargs.get('x_length', 4)
    y_length = kwargs.get('y_length', 4)
    mdc_x_width = kwargs.get('mdc_x_width', 0.3)
    mdc_y_width = kwargs.get('mdc_y_width', 0.3)
    angle_xaxis = kwargs.get('angle_xaxis', 0)
    angle_yaxis = kwargs.get('angle_yaxis', 0)
    cmap = kwargs.get('cmap', 'viridis')
    fontsize = kwargs.get('fontsize', 15)

    k_zero_x = np.array(k_zero_x)
    k_zero_y = np.array(k_zero_y)
    if ax is None or fig is None:
        fig, ax = plt.subplots()

    E = momentum_map.attrs['E']
    E_int = momentum_map.attrs['E_int']
    frame = momentum_map / momentum_map.max()
    frame.plot.imshow(ax=ax, vmin=0, vmax=1, cmap=cmap, add_colorbar=False)
    ax.set_aspect(1)
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(fr'$k_x$ [$\AA^{{-1}}]$', fontsize=fontsize)
    ax.set_ylabel(fr'$k_y$ [$\AA^{{-1}}]$', fontsize=fontsize)
    rect1 = add_rect(k_zero_x[0], x_length, k_zero_x[1], mdc_x_width, ax, edgecolor='lime', facecolor='lime', alpha = 0.2)
    rect2 = add_rect(k_zero_y[0], mdc_y_width, k_zero_y[1], y_length, ax, edgecolor='yellow', facecolor='yellow', alpha = 0.2)

    kx_vals = frame.kx.values
    ky_vals = frame.ky.values
    divider = make_axes_locatable(ax)
    ax_mdc_x = divider.append_axes('top', 1, pad=0, sharex=ax)
    ax_mdc_y = divider.append_axes('right', 1, pad=0, sharey=ax)

    def Gauss(x,x0,FWHM,A):
        sigma = FWHM / 2.355
        return A * np.exp(- (x-x0)**2 /(2 * sigma**2))

    # Interpolator
    interp = RegularGridInterpolator((kx_vals, ky_vals), frame.transpose('kx', 'ky').values, bounds_error=False, fill_value=0)
    
    k_resolution = np.abs(frame.kx.values[0]-frame.kx.values[1])
    N_length = int(np.floor(x_length / k_resolution))
    N_width = int(np.floor(mdc_x_width / k_resolution))
    # Rotate Rectangle Indicator
    transformation_x = mpl.transforms.Affine2D().rotate_deg_around(k_zero_x[0], k_zero_x[1], angle_xaxis) + ax.transData
    rect1.set_transform(transformation_x)
    # Unit vector, and k-line
    kx_unit = np.array([np.cos(np.deg2rad(angle_xaxis)), np.sin(np.deg2rad(angle_xaxis))])
    k_line = np.linspace((-(x_length/2)*kx_unit + k_zero_x), ((x_length/2)*kx_unit + k_zero_x), num=N_length)
    kx_line, ky_line = k_line[:, 0], k_line[:, 1]
    # Normal vector
    normal_vec = np.array([-kx_unit[1], kx_unit[0]])
    w_offsets = np.linspace(-mdc_x_width/2, mdc_x_width/2, num=N_width)
    # Prepare all sampling points
    mdc_x = np.zeros(N_length)
    for i, (kx_i, ky_i) in enumerate(zip(kx_line, ky_line)):
        # Offset coordinates across the perpendicular direction
        kx_offsets = kx_i + normal_vec[0] * w_offsets
        ky_offsets = ky_i + normal_vec[1] * w_offsets

        # Build grid for each offset and energy
        kx_grid = kx_offsets[:, None]
        ky_grid = ky_offsets[:, None]

        pts = np.column_stack([kx_grid.ravel(), ky_grid.ravel()])
        vals = interp(pts).reshape(len(w_offsets))
        mdc_x[i] = np.nanmean(vals, axis=0)

    mdc_x_Ivals = mdc_x
    mdc_x_kvals = np.linspace(-(x_length/2), x_length/2, num=N_length)
    mdc_x = xr.DataArray(mdc_x_Ivals, dims=('k'), coords={'k': mdc_x_kvals}, name='mdc', attrs={'E':E, 'E_int':E_int})

    # Same thing for ky MDC
    N_length = int(np.floor(y_length / k_resolution))
    N_width = int(np.floor(mdc_y_width / k_resolution))
    # Rotate Rectangle Indicator
    transformation_y = mpl.transforms.Affine2D().rotate_deg_around(k_zero_y[0], k_zero_y[1], angle_yaxis) + ax.transData
    rect2.set_transform(transformation_y)
    # Unit vector, and k-line
    ky_unit = np.array([-np.sin(np.deg2rad(angle_yaxis)), np.cos(np.deg2rad(angle_yaxis))])
    k_line = np.linspace((k_zero_y - (y_length/2)*ky_unit), (k_zero_y + (y_length/2)*ky_unit), num=N_length)
    kx_line, ky_line = k_line[:, 0], k_line[:, 1]
    # Normal vector
    normal_vec = np.array([-ky_unit[1], ky_unit[0]])
    w_offsets = np.linspace(-mdc_y_width/2, mdc_y_width/2, num=N_width)
    # Prepare all sampling points
    mdc_y = np.zeros(N_length)
    for i, (kx_i, ky_i) in enumerate(zip(kx_line, ky_line)):
        # Offset coordinates across the perpendicular direction
        kx_offsets = kx_i + normal_vec[0] * w_offsets
        ky_offsets = ky_i + normal_vec[1] * w_offsets

        # Build grid for each offset and energy
        kx_grid = kx_offsets[:, None]
        ky_grid = ky_offsets[:, None]

        pts = np.column_stack([kx_grid.ravel(), ky_grid.ravel()])
        vals = interp(pts).reshape(len(w_offsets))
        mdc_y[i] = np.nanmean(vals, axis=0)

    mdc_y_Ivals = mdc_y
    mdc_y_kvals = np.linspace(-y_length/2, y_length/2, num=N_length)
    mdc_y = xr.DataArray(mdc_y_Ivals, dims=('k'), coords={'k': mdc_y_kvals}, name='mdc')

    colormap = mpl.colormaps['viridis']
    colors = colormap(np.linspace(0.75, 1, 2))

    ax_mdc_x.plot(mdc_x.k.values, mdc_x.values, color=colors[0], alpha=0.7, path_effects=[pe.Stroke(linewidth=3, foreground='black', alpha=0.6), pe.Stroke(foreground='white', alpha=1), pe.Normal()])
    ax_mdc_y.plot(mdc_y.values, mdc_y.k.values, color=colors[1], alpha=0.7, path_effects=[pe.Stroke(linewidth=3, foreground='black', alpha=0.6), pe.Stroke(foreground='white', alpha=1), pe.Normal()])
    
    I_max_x = np.max(mdc_x)
    I_max_y = np.max(mdc_y)

    ax_mdc_x.xaxis.set_tick_params(labelbottom=False, direction='in')
    ax_mdc_x.yaxis.set_tick_params(labelleft=False)
    ax_mdc_x.set_ylim(0, I_max_x*1.05)
    ax_mdc_x.set_yticks([0.5*I_max_x, I_max_x])
    ax_mdc_x.set_title(fr"$E-E_{{\mathrm{{VBM}}}}$ = ({E:.2f} $\pm$ {E_int/2}) eV", fontsize=13)
    ax_mdc_y.yaxis.set_tick_params(labelleft=False, direction='in')
    ax_mdc_y.xaxis.set_tick_params(labelbottom=False)
    ax_mdc_y.set_xlim(0, I_max_y*1.05)
    ax_mdc_y.set_xticks([0.5*I_max_y, I_max_y])
    fig.tight_layout()

    return ax_mdc_x, mdc_x, ax_mdc_y, mdc_y


def add_rect(dim1, dim1_int, dim2, dim2_int, ax, **kwargs):
        edgecolor = kwargs.get("edgecolor", None)
        facecolor = kwargs.get("facecolor", 'grey')
        alpha = kwargs.get("alpha", 0.5)
        linewidth = kwargs.get('linewidth', 0.5)

        rect = (Rectangle((dim1-dim1_int/2, dim2-dim2_int/2), dim1_int, dim2_int , linewidth=linewidth,\
                             edgecolor=edgecolor, facecolor=facecolor, alpha = alpha))
        ax.add_patch(rect) #Add rectangle to plot
        
        return rect

def overlay_bz(shape_type, a, b, ax, color, **kwargs):

    """
    Overlays a custom Brillouin zone polygon on an imshow plot.

    Parameters:
    - shape: 
    - a, b: lattice constants in x and y direction (used to scale Γ, X, Y point labels).
    - ax: matplotlib axes object to draw on.
    - color: color for the polygon edge.
    """

    repeat=kwargs.get("repeat", 0)
    rotation_deg=kwargs.get("rotation_deg", 0)
    fontsize = kwargs.get('fontsize', 14)
    markersize = kwargs.get('markersize', 4)

    def make_rect_bz(a, b):
        X = np.pi / a
        Y = np.pi / b
        return [(-X, -Y), (X, -Y), (X, Y), (-X, Y)]

    def make_hex_bz(a):
        radius = 4 * np.pi / (3 * a)
        angles = np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/6
        return [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    def rotate_shape(coords, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        return [(x*cos_a - y*sin_a, x*sin_a + y*cos_a) for x, y in coords]
        
    # Choose the shape
    if shape_type == 'rectangular':
        base_shape = make_rect_bz(a, b)
        dx, dy = 2 * np.pi / a, 2 * np.pi / b
    elif shape_type == 'hexagonal':
        base_shape = make_hex_bz(a)
        # approximate hexagon repetition spacing
        dx, dy = 4 * np.pi / (3 * a), 4 * np.pi / (3 * a)
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")
    
    # Rotate shape
    shape = rotate_shape(base_shape, rotation_deg)

    # Translate shape to center
    center = (0,0)
    cx, cy = center

    if shape_type == 'hexagonal':
        # Use reciprocal lattice vectors for proper hex tiling
        #b1 = np.array([4 * np.pi / (3 * a), 0])
        #b2 = np.array([-2 * np.pi / (3 * a), 2 * np.pi / (np.sqrt(3) * a)])
        b1 = (2 * np.pi / a) * np.array([1, -1 / np.sqrt(3)])
        b2 = (2 * np.pi / a) * np.array([0, 2 / np.sqrt(3)])
        for i in range(-repeat, repeat + 1):
            for j in range(-repeat, repeat + 1):
                offset = cx + i * b1[0] + j * b2[0], cy + i * b1[1] + j * b2[1]
                translated_shape = [(x + offset[0], y + offset[1]) for x, y in shape]
                patch = Polygon(translated_shape, closed=True, edgecolor=color,
                                facecolor='none', linewidth=2, alpha=0.75)
                ax.add_patch(patch)
                if i == 0 and j == 0 and np.allclose(center, (0, 0)):
                    #ax.plot(0, 0, 'o', markersize=markersize, alpha=0.75, color=color)
                    #ax.text(0.1, 0.1, fr'$\overline{{\Gamma}}$', size=fontsize, color=color)
                    null = 0

    else:
        # Rectangular grid tiling
        for i in range(-repeat, repeat + 1):
            for j in range(-repeat, repeat + 1):
                offset_x = cx + i * dx
                offset_y = cy + j * dy
                translated_shape = [(x + offset_x, y + offset_y) for x, y in shape]
                patch = Polygon(translated_shape, closed=True, edgecolor=color,
                                facecolor='none', linewidth=2, alpha=0.75)
                ax.add_patch(patch)
                if i == 0 and j == 0 and np.allclose(center, (0, 0)):
                    ax.plot(0, 0, 'ko', markersize=4, alpha=0.75)
                    ax.text(0.1, 0.1, fr'$\Gamma$', size=12)

    #bz = Rectangle((0-X, 0-Y), 2*X, 2*Y , linewidth=2, edgecolor=color, facecolor='none', alpha = 0.75)

    #ax.add_patch(bz) #Add bz to plot
    #ax.plot(0,0, 'ko', markersize = 4, alpha = 0.75)
    #ax.plot([0, 0], [Y-0.1, Y+0.1], color = 'black', alpha = 0.75)
    #ax.plot([-X-0.1, -X+0.1], [0, 0], color = 'black', alpha = 0.75)
    #ax.text(-X-0.45, 0, 'X', size=12)
    #ax.text(0, Y+0.15, 'Y', size=12)
    #ax.text(0.1, 0.1, fr'$\Gamma$', size=12)

#I_sum, I_pos, I_pos_sum, I_neg, I_neg_sum = get_data_chunks([-180,-100], t0, ax_delay_offset) #Get the Neg and Pos delay time arrays

def CircleMasks(I_res, k, radius, repeat=1, **kwargs):

    invert = kwargs.get('invert', False)
    
    kx, ky = k
    angle_step = (2*np.pi) / repeat
    points = []
    for i in range(repeat):
        angle = i * angle_step
        x = kx * np.cos(angle) - ky * np.sin(angle)
        y = kx * np.sin(angle) + ky * np.cos(angle)
        points.append([x, y])
    point_check = []
    for point in points:
        point_check.append(((I_res.kx - point[0])**2 + (I_res.ky - point[1])**2) < radius**2)
    if repeat ==1:
        total_check = point_check[0]
    else:
        total_check = np.copy(point_check[0])
        for i in range(repeat-1):
            total_check = np.logical_or(total_check, point_check[i+1])

    if invert:
        total_check = np.logical_not(total_check)
    masked = I_res.where(total_check, other=0)
    return masked

def custom_colormap(CMAP, lower_portion_percentage):
    # create a colormap that consists of
    # - 1/5 : custom colormap, ranging from white to the first color of the colormap
    # - 4/5 : existing colormap
    
    # set upper part: 4 * 256/4 entries
    CMAP = plt.get_cmap(CMAP)
    upper =  CMAP(np.arange(256))
    upper = upper[56:,:]
    #upper = upper[0:,:]

    # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
    lower_portion = int(1/lower_portion_percentage) - 1
    
    lower = np.ones((int(200/lower_portion),4))
    # - modify the first three columns (RGB):
    #   range linearly between white (1,1,1) and the first color of the upper colormap
    for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
    
    # combine parts of colormap
    cmap = np.vstack(( lower, upper ))
    
    # convert to matplotlib colormap
    custom_cmap = mpl.colors.ListedColormap(cmap, name='custom', N=cmap.shape[0])
    
    return custom_cmap

def create_custom_diverging_colormap(map1, map2):
    # Create the negative part from seismic (from 0.5 to 1 -> blue to white)
    seismic = plt.get_cmap(map1)
    seismic_colors = seismic(np.linspace(0, 1, 128))  # -1 to 0

    # Create the positive part from viridis (from 0 to 1)
    viridis = plt.get_cmap(map1)
    viridis = cmap_LTL
    viridis_colors = viridis(np.linspace(0, 1, 128))  # 0 to 1

    # Combine both
    combined_colors = np.vstack((seismic_colors[::-1], viridis_colors))

    # Create a new colormap
    custom_colormap = LinearSegmentedColormap.from_list('seismic_viridis', combined_colors)

    return custom_colormap

#%% Functions For Fitting Data: Time Traces

def monoexp(t, A, tau):
    return A * np.exp(-(t) / tau) * (t >= 0)  # Ensure decay starts at t=t0

def monoexp_const(t, A, tau, c):
    return (A * np.exp(-t/tau) + c) * (t >= 0)

# Define the biexponnential decay function (Exciton)    
def biexp(t, A, tau1, B, tau2):
    return ( A * np.exp(-t / tau1) + B * np.exp(-t / tau2))  * (t >= 0)  # Ensure decay starts at t=0

# Define the conduction band model: exponential rise + decay
def exp_rise_monoexp_decay(t, A, tau_rise, tau_decay1):
    return A * (1 - np.exp(-t / tau_rise)) * (np.exp(-t / tau_decay1)) * (t >= 0)

def exp_rise_biexp_decay(t, A, tau_rise, D, tau_decay1, tau_decay2):
    return A * (1 - np.exp(-t / tau_rise)) * (D * np.exp(-t / tau_decay1) + (1-D) * np.exp(-t / tau_decay2)) * (t >= 0)

# Define the Instrumental Response Function (IRF) as a Gaussian
def IRF(t, sigma_IRF):
    return np.exp(-t**2 / (2 * sigma_IRF**2)) / (sigma_IRF * np.sqrt(2 * np.pi))

# Convolution of the signal with the IRF
def convolved_signal_1(t, signal_function, sigma_IRF, *params):
    dt = np.mean(np.diff(t))  # Time step
    signal = signal_function(t, *params)  # Compute signal
    irf = IRF(t - t[len(t)//2], sigma_IRF)  # Shift IRF to center
    irf /= np.sum(irf) * dt  # Normalize IRF
    convolved = fftconvolve(signal, irf, mode='same') * dt  # Convolve with IRF
    return convolved

def convolved_signal(t, signal_function, sigma_IRF, *params):
    dt = np.mean(np.diff(t))

    # Extend the time axis on both sides to avoid edge effects
    pad_width = int(5 * sigma_IRF / dt)  # enough padding for Gaussian tail
    t_pad = np.linspace(t[0] - pad_width * dt, t[-1] + pad_width * dt, len(t) + 2 * pad_width)

    # Evaluate signal on the extended time axis
    signal_ext = signal_function(t_pad, *params)

    # Create centered Gaussian IRF
    irf = np.exp(-((t_pad - np.median(t_pad)) ** 2) / (2 * sigma_IRF ** 2))
    irf /= np.sum(irf) * dt  # Normalize area under the IRF to 1

    # Convolve using FFT
    conv_ext = fftconvolve(signal_ext, irf, mode='same') * dt

    # Trim back to original t range
    convolved = conv_ext[pad_width : -pad_width]
    
    return convolved

def make_convolved_model(base_model, t, sigma_IRF):
    """
    Returns a callable f(t, *params) which evaluates the convolved model at time t.
    """
    dt = np.mean(np.diff(t))
    pad_width = int(5 * sigma_IRF / dt)
    t_pad = np.linspace(t[0] - pad_width * dt, t[-1] + pad_width * dt, len(t) + 2 * pad_width)

    # Centered Gaussian IRF
    irf = np.exp(-((t_pad - np.median(t_pad)) ** 2) / (2 * sigma_IRF ** 2))
    irf /= np.sum(irf) * dt

    def model(t_fit, *params):
        signal_ext = base_model(t_pad, *params)
        conv_ext = fftconvolve(signal_ext, irf, mode='same') * dt
        return conv_ext[pad_width : -pad_width]

    return model

model_dict = {
        'monoexp': monoexp,
        'exp_rise_monoexp_decay': exp_rise_monoexp_decay,
        'biexp': biexp,
        'exp_rise_biexp_decay': exp_rise_biexp_decay,
        'monoexp_const': monoexp_const
}


def fit_time_trace(fit_model, delay_axis, time_trace, p0, bounds, convolve=False, sigma_IRF=None):
    """
    Fit a time trace using a specified model, optionally convolved with an IRF.

    Parameters:
    - fit_model (str): Name of the model ('monoexp', 'exp_rise_monoexp_decay')
    - delay_axis (array): Time delay values
    - time_trace (array): Measured time trace
    - p0 (tuple/list): Initial guess for fit parameters
    - bounds (2-tuple): Bounds for fit parameters ((lower_bounds), (upper_bounds))
    - convolve (bool): Whether to convolve the model with an IRF
    - sigma_IRF (float): Width of the Gaussian IRF (if convolve=True)

    Returns:
    - popt: Optimal parameters from curve_fit
    - pcov: Covariance of the parameters
    - fit_curve: Evaluated fit curve
    """

    if fit_model not in model_dict:
        raise ValueError(f"Unsupported model: {fit_model}")

    base_model = model_dict[fit_model]

    #if convolve:
    #    if sigma_IRF is None:
    #        raise ValueError("sigma_IRF must be provided if convolve=True")
    #    def model_func(t, *params):
    #        return convolved_signal(t, base_model, sigma_IRF, *params)
    #else:
    #    model_func = base_model

    if convolve:
        model_func = make_convolved_model(base_model, delay_axis, sigma_IRF)
    else:
        model_func = base_model

    popt, pcov = curve_fit(model_func, delay_axis, time_trace, p0=p0, bounds=bounds)
    fit_curve = model_func(delay_axis, *popt)

    return popt, pcov, fit_curve


import numpy as np

def print_fit_results(model_name, popt, pcov):
    """
    Print fit parameters and uncertainties based on model name.
    """

    def build_param_list(popt, perr, param_names):
        
        return {
            name: val for name, val in zip(param_names, popt)
        } | {
            "errors": {name: err for name, err in zip(param_names, perr)}
        }

    model_param_names = {
        'monoexp': ['A', 'tau_decay1'],
        'biexp': ['A', 'tau_decay1', 'B', 'tau_decay2'],
        'exp_rise_monoexp_decay': ["A", 'tau_rise', 'tau_decay1'],
        'exp_rise_biexp_decay': ['A', 'tau_rise', 'D', 'tau_decay1', 'tau_decay2'],
        'monoexp_const' : ['A', 'tau_decay1', 'c']
        # Add more models here as needed
    }

    if model_name not in model_param_names:
        raise ValueError(f"Unsupported model: {model_name}")

    param_names = model_param_names[model_name]
    errors = np.sqrt(np.diag(pcov))

    params_list = build_param_list(popt, errors, param_names)

    if model_name == 'monoexp':
        plot_label = fr"$\tau_{1}$ = {params_list['tau_decay1']:.3f} fs"
    elif model_name == 'exp_rise_monoexp_decay':
        plot_label = fr"$\tau_{{r}}$ = {params_list['tau_rise']:.3f} fs, $\tau_{1}$ = {params_list['tau_decay1']:.3f} fs"
    elif model_name == 'biexp':
        plot_label = fr"$\tau_{1}$ = ({params_list['tau_decay1']:.3f}$\pm${params_list['errors']['tau_decay1']:.3f}) fs {'\n'}$\tau_{2}$ = {params_list['tau_decay2']:.3f} fs"
    elif model_name == 'exp_rise_biexp_decay':
        plot_label = fr"$\tau_{{r}}$ = {params_list['tau_rise']:.3f} fs, $\tau_{1}$ = {params_list['tau_decay1']:.3f} fs, $\tau_{2}$ = {params_list['tau_decay2']:.3f} fs"
    elif model_name == 'monoexp_const':
        plot_label = fr'$\tau_{1}$ = {params_list['tau_decay1']:.3f} fs'

    print(f"\nFit Results for model: {model_name}")
    print("-" * 40)
    for name, val, err in zip(param_names, popt, errors):
        print(f"{name:10s} = {val:10.2f} ± {err:8.2f} ({100*err/val:6.2f} %)")
    print("-" * 40)
    
    return (params_list, plot_label)

cmap_LTL = custom_colormap('viridis', 0.2)
cmap_LTL2 = create_custom_diverging_colormap('Reds', 'viridis')

# %%
