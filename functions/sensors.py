"""
This module contain several functions required for the computations in the notebooks
main
load_modes_plot_transmission
"""
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import scipy as sp
from scipy.optimize import fsolve
from mpmath import mp
from IPython.display import clear_output
import os
import sys
import time
from IPython.display import display, clear_output
import matplotlib as mpl
from derivative import derivative
from gold_permittivity import gold_permittivity
from silica_permittivity import silica_permittivity
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.io import savemat
from scipy.io import loadmat



global e0 
global c0
e0 = 8.85418787162e-12 # permittivity of free space
c0 = 299792458 # speed of light
# matrix to solve from simple slab model 
def disp_slab_5_mat(neff,*data):
    """
    dispersion equation for the "three slab" TM mode
    polarization: TM
    slabs of thickness di and material permittivity e_i and thickness d_i are distributed along x as follows:
    thickness and permittivity:    
    infinite    | d1 | d2 | d3 | infinite 
    ------------|----|----|----|----------
    e1          | e2 | e3 | e4 | e5       
    ---------------------------------------> x 
    y is "out of plane" (1D mode, so it's infinite): this is a TM mode, so we have only Hy 
    x is perpendicular to the interface: this is TM mode, se Ex is the dominant electric field component
    z is the direction of propagation: this is a TM mode, so we have an Ez component

    --------
    neff          :     effective index, independent variable -- neff is a solution if the output of this function is zero     
    *data         :     k0, e1, e2, e3, e4, e5, d1, d2, d3          
                            
    Returns
    -------
    dispersion_matrix    :       complex ndarray
                                dispersion quation
    Reference
    -------
    This is the generalization of a textbook case, where we express TM fields / boundary conditions, set up a matrix, solve - see for example:
    "Theory of Dielectric Optical Waveguides" by Dietrich Marcuse
    """
    k0, e1, e2, e3, e4, e5, d1, d2, d3 = data
    g1 = k0*np.sqrt(neff**2-e1)
    k2 = k0*np.sqrt(e2-neff**2)
    k3 = k0*np.sqrt(e3-neff**2)
    k4 = k0*np.sqrt(e4-neff**2)
    g5 = k0*np.sqrt(neff**2-e5)
    dispersion_matrix = np.matrix([[1, -1, -1, 0, 0, 0, 0, 0],
                                   [0, np.exp(1j*k2*d1), np.exp(-1j*k2*d1), -np.exp(1j*k3*d1), -np.exp(-1j*k3*d1), 0, 0, 0],
                                   [0, 0, 0, np.exp(1j*k3*(d1+d2)), np.exp(-1j*k3*(d1+d2)), -np.exp(1j*k4*(d1+d2)), -np.exp(-1j*k4*(d1+d2)), 0],
                                   [0, 0, 0, 0, 0, np.exp(1j*k4*(d1+d2+d3)), np.exp(-1j*k4*(d1+d2+d3)),-np.exp(-g5*(d1+d2+d3))],
                                   [-1j*g1/(k0*c0*e0*e1), -k2/(k0*c0*e0*e2), k2/(k0*c0*e0*e2), 0, 0, 0, 0, 0],
                                   [0, k2/(k0*c0*e0*e2)*np.exp(1j*k2*d1), -k2/(k0*c0*e0*e2)*np.exp(-1j*k2*d1), -k3/(k0*c0*e0*e3)*np.exp(1j*k3*d1), k3/(k0*c0*e0*e3)*np.exp(-1j*k3*d1),0,0,0],
                                   [0, 0, 0, k3/(k0*c0*e0*e3)*np.exp(1j*k3*(d1+d2)), -k3/(k0*c0*e0*e3)*np.exp(-1j*k3*(d1+d2)), -k4/(k0*c0*e0*e4)*np.exp(1j*k4*(d1+d2)), k4/(k0*c0*e0*e4)*np.exp(-1j*k4*(d1+d2)), 0],
                                   [0, 0, 0, 0, 0, k4/(k0*c0*e0*e4)*np.exp(1j*k4*(d1+d2+d3)), -k4/(k0*c0*e0*e4)*np.exp(-1j*k4*(d1+d2+d3)),-1j*g5*np.exp(-g5*(d1+d2+d3))/(k0*c0*e0*e5)]])
    out = dispersion_matrix
    return out

# dispersion relation comes from determinant of above matrix
def disp_slab_5(neff,*data):
    """
    determinant of the dispersion equation for the "three slab" TM mode
    --------
    neff          :     effective index, independent variable -- neff is a solution if the output of this function is zero     
    *data         :     k0, e1, e2, e3, e4, e5, d1, d2, d3          
                        complex
                            
    Returns
    -------
    out    :            complex ndarray
                        determinant of dispersion quation
    """
    k0, e1, e2, e3, e4, e5, d1, d2, d3 = data
    out = np.linalg.det(disp_slab_5_mat(neff,*data))
    return out

# function to use for finding zero of dispersion relation numerically
def disp_slab_5_solve(neff_array,*data):
    """
    separate real and imaginary parts of the determinant of the dispersion equation for the "three slab" TM mode
    --------
    neff_array          :     effective index
    *data               :     k0, e1, e2, e3, e4, e5, d1, d2, d3          
                            
    Returns
    -------
    out    :            complex ndarray
                        (real, imaginary) determinant of dispersion quation
    """
    neff = neff_array[0]+1j*neff_array[1]
    det_dispersion_matrix = disp_slab_5(neff,*data)
    out = np.array((np.real(det_dispersion_matrix),np.imag(det_dispersion_matrix)))
    return out

# function to obtain the coefficients using Cramer's rule
def coefficient(M,l,n):
    M=np.matrix(M)
    M_list=np.array(M.tolist())
    detM = np.linalg.det(M)
    l = mp.matrix(l)
    l_list = np.array(l.tolist())
    M_list[:,n]=l_list[:,0]
    out = np.linalg.det(np.matrix(M_list))/detM
    return out

# define a function for permittivity distributions
def epsilon_distribution(e1,e2,e3,e4,e5,d1,d2,d3,x):
    e = np.zeros(x.size)+0*1j
    e[np.where(x<0)] = e1
    e[np.where(np.logical_and(x>=0, x<=d1))] = e2
    e[np.where(np.logical_and(x>d1, x<d1+d2))] = e3
    e[np.where(np.logical_and(x>=d1+d2, x<=0+d1+d2+d3))] = e4
    e[np.where(x>d1+d2+d3)] = e5
    return e

def calculate_fields(d1,d2,d3,dx,neff0,k0,e1,e2,e3,e4,e5):
    w = k0*c0
# get the coefficients from the neff, Cramer's rule... start from A=1 to get the rest, renormalize later.
    dispersion_matrix = disp_slab_5_mat(neff0,k0, e1, e2, e3, e4, e5, d1, d2, d3)
    dispersion_matrix_list = np.array(dispersion_matrix.tolist())
    dispersion_matrix_list_red=np.delete(np.delete(dispersion_matrix_list, 0, 1),7,0) # reduce matrix
    l = -dispersion_matrix_list[0:7,0]
    A = 1
    B = coefficient(dispersion_matrix_list_red,l,0)
    C = coefficient(dispersion_matrix_list_red,l,1)
    D = coefficient(dispersion_matrix_list_red,l,2)
    E = coefficient(dispersion_matrix_list_red,l,3)
    F = coefficient(dispersion_matrix_list_red,l,4)
    G = coefficient(dispersion_matrix_list_red,l,5)
    H = coefficient(dispersion_matrix_list_red,l,6)
    
#   calculate the ki...
    g1 = k0*np.sqrt(neff0**2-e1)
    k2 = k0*np.sqrt(e2-neff0**2)
    k3 = k0*np.sqrt(e3-neff0**2)
    k4 = k0*np.sqrt(e4-neff0**2)
    g5 = k0*np.sqrt(neff0**2-e5)
#   field is distributed in x...
    x1 = np.arange(-1e-6,0-dx,dx)
    x2 = np.arange(0,d1,dx)
    x3 = np.arange(d1+dx,d1+d2,dx)
    x4 = np.arange(d1+d2+dx,d1+d2+d3,dx)
    x5 = np.arange(d1+d2+d3+dx,(d1+d2+d3)+1e-6,dx)
    x_tot =  (np.concatenate([x1,x2,x3,x4,x5]))

#   magnetic field in y
    Hy1 = A*np.exp(g1*x1)
    Hy2 = B*np.exp(1j*k2*x2)+C*np.exp(-1j*k2*x2)
    Hy3 = D*np.exp(1j*k3*x3)+E*np.exp(-1j*k3*x3)
    Hy4 = F*np.exp(1j*k4*x4)+G*np.exp(-1j*k4*x4)
    Hy5 = H*np.exp(-g5*x5)
    Hy_tot = (np.concatenate([Hy1,Hy2,Hy3,Hy4,Hy5]))
    
    # define permittivity distribution
    e = epsilon_distribution(e1,e2,e3,e4,e5,d1,d2,d3,x_tot)
    
    # electric field Ex
    b=neff0*k0
    Ex_tot = -b/w/e/e0*Hy_tot

    # normalize 
    norm = 0.5*(np.trapz(-Ex_tot*(Hy_tot),x=x_tot))

    # make sure all fields consistent
    Hy_tot = Hy_tot/np.sqrt(norm)
    Ex_tot = Ex_tot/np.sqrt(norm)

    # including Ez for completeness, although it doesn't get used...
    dHy_dx = derivative(x_tot,Hy_tot)
    Ez_tot = -1j/w/e/e0*(dHy_dx)
    return x_tot, Hy_tot, Ex_tot, Ez_tot

# finding the modes and fields in one function
def mode_solver(lm,neff_guess,e1,e2,e3,e4,e5,d1,d2,d3,dx):
    k0 = 2*np.pi/lm  
    neff_fsolve = fsolve(disp_slab_5_solve,np.array([np.real(neff_guess),np.imag(neff_guess)+1e-5]),args=(k0, e1, e2, e3, e4, e5, d1, d2, d3),xtol = 1e-13)
    neff = neff_fsolve[0]+1j*neff_fsolve[1]
    x, Hy, Ex, Ez = calculate_fields(d1,d2,d3,dx,neff,k0,e1,e2,e3,e4,e5)
    return x, Hy, Ex, Ez, neff

def show_dispersion_equation(neffr,neffi, k0, e1, e2, e3, e4, e5, d1, d2, d3):
    neffR, neffI = np.meshgrid(neffr,neffi)
    neff = neffR+neffI*1j
    dispersion_function=np.zeros([neffr.size,neffi.size])+0*1j

    #  calculate the dispersion relation in a region of interest, we will look for the zeros
    for kx in range(0,neffr.size):
        clear_output(wait=True)
        print(str(float(kx+1)/float(neffr.size)*100)+"% complete")
        sys.stdout.flush()
        for ky in range(0,neffi.size):
            dispersion_function[kx,ky] = disp_slab_5(neff[kx,ky],k0, e1, e2, e3, e4, e5, d1, d2, d3)

    # plot dispersion function, we look for zeros
    plt.figure()
    plt.pcolor(neffR,neffI,10*np.log10(np.abs(dispersion_function)))
    plt.xlabel('Real(neff)')
    plt.ylabel('Real(neff)')
    plt.title('Dispersion Equation [Log10]')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
# finding the modes and fields in one function
def mode_solver_no_fields(lm,neff_guess,e1,e2,e3,e4,e5,d1,d2,d3):
    k0 = 2*np.pi/lm  
    neff_fsolve = fsolve(disp_slab_5_solve,np.array([np.real(neff_guess),np.imag(neff_guess)+1e-5]),args=(k0, e1, e2, e3, e4, e5, d1, d2, d3),xtol = 1e-13)
    neff = neff_fsolve[0]+1j*neff_fsolve[1]
#    x, Hy, Ex, Ez = calculate_fields(d1,d2,d3,dx,neff,k0,e1,e2,e3,e4,e5)
    return neff

def mode_solver_plot(first_wavelength,neff_guess,e1,e2,e3,e4,e5,d1,d2,d3,dx):
    # lossy example
    x, Hy, Ex, Ez, neff0 = mode_solver(first_wavelength,neff_guess,e1,e2,e3,e4,e5,d1,d2,d3,dx)

    # lossless example - gold film is real
    #x, Hy, Ex, Ez, neff0 = mode_solver(wavelength,neff_guess,e1,e2,e3,np.real(e4),e5,d1,d2,d3,dx)

    plt.figure()
    plt.plot(x/1e-6,np.real(Hy),label='real')
    plt.plot(x/1e-6,np.imag(Hy),label='imag')
    plt.xlim([-1, 1])
    plt.legend()
    plt.xlabel('x [um]')
    plt.ylabel('Hy')
    plt.title('neff = ' + str(neff0) )
    plt.show()
    
    return x, Hy, Ex, Ez, neff0

def get_permittivity(lm,n_analyte,select_configuration):
    if select_configuration=='dielectric isolated':     
        e1,e2,e3,e4,e5 = silica_permittivity(lm), 3.5**2, silica_permittivity(lm), silica_permittivity(lm), n_analyte**2
    if select_configuration=='plasmonic isolated lossy':     
        e1,e2,e3,e4,e5 = silica_permittivity(lm), silica_permittivity(lm), silica_permittivity(lm), gold_permittivity(lm), n_analyte**2
    if select_configuration=='plasmonic isolated lossless':     
        e1,e2,e3,e4,e5 = silica_permittivity(lm), silica_permittivity(lm), silica_permittivity(lm), np.real(gold_permittivity(lm)), n_analyte**2
    if select_configuration=='supermode lossy':     
        e1,e2,e3,e4,e5 = silica_permittivity(lm), 3.5**2, silica_permittivity(lm), gold_permittivity(lm), n_analyte**2
    if select_configuration=='supermode lossless':     
        e1,e2,e3,e4,e5 = silica_permittivity(lm), 3.5**2, silica_permittivity(lm), np.real(gold_permittivity(lm)), n_analyte**2
    return e1,e2,e3,e4,e5

def loop_over_analytes_and_wavelength(wavelength_range,analyte_index_range,neff_array,d1,d2,d3,select_configuration):
    for ka in range(0,len(analyte_index_range)):
        clear_output(wait=True)
        print("looping over analyte index... " + str(float(ka+1)/float(len(analyte_index_range))*100)+"% complete")

        # define analyte index as a separate variable at the top for clarity

        n_analyte = analyte_index_range[ka]

        # if you move on to a new analyte index,  obtain the guess value from previous analyte at the first wavelength  
        
        if ka>0:
            lm = wavelength_range[0]
            k0 = 2*np.pi/lm
            e1,e2,e3,e4,e5 = get_permittivity(lm,n_analyte,select_configuration)
            neff_array[ka,0] = mode_solver_no_fields(wavelength_range[0],neff_array[ka-1,0],e1,e2,e3,e4,e5,d1,d2,d3)
            
        # ...then loop over every other wavelength  
        for kn in range(1,len(wavelength_range)):
            lm = wavelength_range[kn]
            k0 = 2*np.pi/lm
            e1,e2,e3,e4,e5 = get_permittivity(lm,n_analyte,select_configuration)
            neff_array[ka,kn] = mode_solver_no_fields(wavelength_range[kn],neff_array[ka,kn-1],e1,e2,e3,e4,e5,d1,d2,d3)
    return neff_array

def get_kappa_and_neff_predicted_by_CMT(wavelength_range,neff_0,neff_s,neff_s_ll,neff_1,neff_2,neff_1_ll,neff_2_ll):
    B0 = 2*np.pi/wavelength_range*neff_0 # dielectric, isolated
    BS = 2*np.pi/wavelength_range*neff_s # plasmonic, isolated lossy
    BS_ll = 2*np.pi/wavelength_range*neff_s_ll # plasmonic, isolated lossless

    B1 = 2*np.pi/wavelength_range*neff_1 # supermode 1, lossy
    B2 = 2*np.pi/wavelength_range*neff_2 # supermode 2, lossy
    B1_ll = 2*np.pi/wavelength_range*neff_1_ll # supermode 1, lossless
    B2_ll = 2*np.pi/wavelength_range*neff_2_ll # supermode 2, lossless

    # get \kappa from isolated modes
    D = (B1_ll-B2_ll)/2     # D is \tilde{\Delta} in Eq. (8) of the paper
    d_ll = (B0-BS_ll)/2     # d_ll is \Delta in Eq. (8)  of the paper (lossless modes)
    kappa = np.sqrt(D**2-d_ll**2)  # C is \kappa in Eq. (8)  of the paper

    # apply the above \kappa to get the EMs predicted by CMT
    Dav = (B0+BS)/2         # Dav is \overline{\beta} - use lossy modes
    d = (B0-BS)/2           # d is \Delta in Eq. (8) of the paper - use lossy modes
    # calculate the EigenMode propagation constants predicted by CMT, for later comparison with "exact" propagation constants 
    B1_CMT = Dav+np.sqrt(kappa**2+d**2) 
    B2_CMT = Dav-np.sqrt(kappa**2+d**2) 

    neff_1_CMT = B1_CMT/(2*np.pi/wavelength_range)
    neff_2_CMT = B2_CMT/(2*np.pi/wavelength_range)
    return kappa, neff_1_CMT, neff_2_CMT

def plot_Fig_6(wavelength_range,analyte_index_range,neff_0,neff_s,neff_s_ll,neff_1,neff_2,neff_1_ll,neff_2_ll,kappa, neff_1_CMT, neff_2_CMT):
    EP_exact = np.abs((neff_1-neff_2)) # Eq. 10
    EP_CMT = np.abs(np.imag(neff_s)/2-np.real(kappa/(2*np.pi/wavelength_range)))+np.abs(np.real(neff_0-neff_s)) # Eq. 11

    [LM,NA] = np.meshgrid(wavelength_range,analyte_index_range)

    plt.figure(figsize=(8,2), dpi=300)
    plt.subplot(1,2,1)
    plt.pcolor(LM/1e-9,NA,np.log10(np.abs(EP_CMT)))
    plt.title('CMT EP condition')
    plt.xlabel('wavelength [nm]')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.pcolor(LM/1e-9,NA,10*np.log10(np.abs(EP_exact)))
    plt.title('"exact" EP condition')
    plt.xlabel('wavelength [nm]')
    plt.colorbar()
    plt.show()
    
    return EP_exact, EP_CMT

def get_and_plot_characteristic_lengths(analyte_index_range,wavelength_range,neff_0,neff_s,neff_1,neff_2):
    # allocate variables

    lm_pm_lossy = np.zeros(len(analyte_index_range)) # phase matching wavelength, lossy
    lm_lm_lossy= np.zeros(len(analyte_index_range)) # intersection of imaginary parts, lossy

    beat_length_exact = np.zeros(len(analyte_index_range)) # beat length according to exact solution
    beat_length_CMT = np.zeros(len(analyte_index_range)) # beat length according to CMT

    loss_length_1 =  np.zeros(len(analyte_index_range)) # loss length first mode according to exact solution
    loss_length_2 =  np.zeros(len(analyte_index_range)) # loss length second mode according to exact solution

#    loss_length_1_CMT =  np.zeros(len(analyte_index_range)) # loss length first mode according to CMT
#    loss_length_2_CMT =  np.zeros(len(analyte_index_range)) # loss length second mode according to CMT

    for ka in range(0,len(analyte_index_range)):
        pm_index_lossy_temp = np.where(np.abs(np.real(neff_0[ka,:])-np.real(neff_s[ka,:]))==np.min(np.abs(np.real(neff_0[ka,:])-np.real(neff_s[ka,:]))))     
        pm_index_lossy = int(pm_index_lossy_temp[0])
    
        lm_pm_lossy[ka] = wavelength_range[pm_index_lossy] # phase matching wavelength
    
        lm_index_lossy_temp = np.where(np.abs(np.imag(neff_1[ka,:])-np.imag(neff_2[ka,:]))==np.min(np.abs(np.imag(neff_1[ka,:])-np.imag(neff_2[ka,:]))))
        lm_index_lossy = int(lm_index_lossy_temp[0][0])

        lm_lm_lossy[ka] = wavelength_range[lm_index_lossy] # intersection of imaginary parts
    
        # beat lengths are taken at phase matching (lossy)
        beat_length_exact[ka] = wavelength_range[pm_index_lossy]/(np.abs(np.real(neff_1[ka,pm_index_lossy])-np.real(neff_2[ka,pm_index_lossy])))/2
        # beat_length_CMT[ka] = wavelength_range[pm_index_lossy]/(np.abs(np.real(neff_1_CMT[ka,pm_index_lossy])-np.real(neff_2_CMT[ka,pm_index_lossy])))/2

        # loss length are taken at phase matching (lossy)
        loss_length_1[ka] = 1/(4*np.pi/wavelength_range[pm_index_lossy]*np.imag(neff_1[ka,pm_index_lossy]))
        loss_length_2[ka] = 1/(4*np.pi/wavelength_range[pm_index_lossy]*np.imag(neff_2[ka,pm_index_lossy]))
    
        # CMT
        #loss_length_1_CMT[ka] = 1/(4*np.pi/wavelength_range[pm_index_lossy]*np.imag(neff_1_CMT[ka,pm_index_lossy]))
        #loss_length_2_CMT[ka] = 1/(4*np.pi/wavelength_range[pm_index_lossy]*np.imag(neff_2_CMT[ka,pm_index_lossy]))
        
    plt.figure(figsize=(10, 4), dpi=100)
    plt.subplot(1,2,1)
    plt.plot(analyte_index_range,beat_length_exact/1e-6)
    plt.plot([1.3,1.442],[10,10],'--k')
    plt.plot([1.442,1.442],[0,10],'--k')
    plt.ylim([2,12])
    plt.xlim([1.35,1.45])
    plt.xlabel('analyte index [RIU]')
    plt.ylabel('beat length [um]')
    plt.subplot(1,2,2)
    plt.plot(analyte_index_range,(loss_length_1+loss_length_2)/1e-6/2)
    plt.plot(analyte_index_range,loss_length_1/1e-6,'--k')
    plt.plot(analyte_index_range,loss_length_2/1e-6,'--k')
    plt.ylim([1.5,4])
    plt.xlim([1.35,1.45])
    plt.ylabel('absorption length [um]')
    plt.xlabel('analyte index [RIU]')
    plt.show()

    return loss_length_1, loss_length_2, beat_length_exact, lm_pm_lossy

def sensitivity_fit_and_plot(na_min,na_max,analyte_index_range, lm_pm_lossy):
    na_range_plot = np.linspace(na_min,na_max,1000)
    
    ka_min =    int(np.where(np.abs(np.min(na_range_plot)-analyte_index_range)==np.min(np.abs(np.min(na_range_plot)-analyte_index_range)))[0])
    ka_max =    int(np.where(np.abs(np.max(na_range_plot)-analyte_index_range)==np.min(np.abs(np.max(na_range_plot)-analyte_index_range)))[0])
    
    index_max = int(np.where(np.abs(na_min-analyte_index_range)==np.min(np.abs(na_min-analyte_index_range)))[0])
    index_min = int(np.where(np.abs(na_max-analyte_index_range)==np.min(np.abs(na_max-analyte_index_range)))[0])

    z = np.polyfit(analyte_index_range[ka_max:ka_min], lm_pm_lossy[ka_max:ka_min], 1)
    pi = np.poly1d(z)
    sensitivity_lm_pm_lossy_poly = np.real(derivative(na_range_plot,pi(na_range_plot)))
    
    plt.figure(figsize=(11, 4), dpi=100)
    plt.subplot(1,2,1)
    plt.plot(analyte_index_range[range(index_min,index_max)],lm_pm_lossy[range(index_min,index_max)]/1e-9,'--o',label= 'real parts cross (isolated)')
    plt.plot(na_range_plot,na_range_plot*sensitivity_lm_pm_lossy_poly/1e-9-max(na_range_plot*sensitivity_lm_pm_lossy_poly/1e-9)+np.max(lm_pm_lossy[range(index_min,index_max)])/1e-9,'--k',label='fit')
    plt.ylim([1500,1900])
    plt.xlim([na_min,na_max])
    plt.gca().set_prop_cycle(None)
    plt.legend()
    plt.xlabel('analyte index')
    plt.ylabel('lambda R')
    plt.subplot(1,2,2)
    plt.plot(na_range_plot,sensitivity_lm_pm_lossy_poly/1e-9)
    plt.ylim([-2500,-2000])
    plt.xlim([na_min,na_max])
    plt.xlabel('analyte index')
    plt.ylabel('sensitivity [nm/RIU]')
    plt.show()
    print('Sensitivity is ' + str(np.mean(sensitivity_lm_pm_lossy_poly)/1e-9) + 'nm/RIU')

    return na_range_plot, sensitivity_lm_pm_lossy_poly

def coupled_mode_theory(z_CMT,beta_dielectric_isolated,beta_plasmonic_isolated_lossy,coupling_coefficient_lossless):

    global B1_CMT, B2_CMT, C_CMT # define global variables to be used in CMT 

    C_CMT =  coupling_coefficient_lossless # coupling coefficient from real part calculation
    B1_CMT = beta_dielectric_isolated # dielectric mode
    B2_CMT = beta_plasmonic_isolated_lossy # plasmonic mode (lossy)

    # this is Eq. (9) in the paper    
    def vdp1(z_CMT, y):
        return np.array([(C_CMT*y[1] - B1_CMT*y[0])*(-1j), (C_CMT*y[0] - B2_CMT*y[1])*(-1j)])

    z0, z1 = 0, np.max(z_CMT)                  # start and end
    y0 = [1+0*1j, 0+0*1j]                  # initial value
    y = np.zeros((len(z_CMT), len(y0)))+0*1j   # array for solution
    y[0, :] = [1+0*1j, 0+0*1j]             # initial value in start of array

        # integrate complex differential equation
    r = integrate.complex_ode(vdp1).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, z0)   # initial values
    for i in range(1, z_CMT.size):
        y[i, :] = r.integrate(z_CMT[i]) # get one more value, add it to the array
        if not r.successful():
            raise RuntimeError("Could not integrate")

    T_dielectric = np.abs(y[:,0])**2 # power in dielectric mode
    
    return(T_dielectric)

def load_comsol_data():
    temp= loadmat("data/sensor_data_400nm_sep.mat")
    L_comsol = temp['L_range'][0]
    T_comsol = temp['T']
    na_comsol = temp['na_range'][0]
    lm_comsol = temp['lm_reduced'][0]

    temp= loadmat("data/sensor_data_400nm_50um_finer.mat")
    T_50um = temp['T']

    T_comsol = np.array([ T_comsol[:,0,:], T_comsol[:,1,:],  T_comsol[:,2,:], T_comsol[:,3,:],T_50um[:,0,:]])
    L_comsol = np.append(L_comsol,50e-6)

    return lm_comsol, na_comsol, L_comsol, T_comsol

def plot_CMT_COMSOL(lm_comsol,na_comsol,L_comsol,T_comsol,wavelength_range,analyte_index_range,z,T_CMT):

    for L_select in L_comsol[range(1,len(L_comsol))]:   

        kl = (np.where(np.abs(L_select-z)==np.min(np.abs(L_select-z))))[0][0]
        klc = (np.where(np.abs(L_select-L_comsol)==np.min(np.abs(L_select-L_comsol))))[0][0]

        [LMC, NAC] = np.meshgrid(lm_comsol,na_comsol)
        [LM,NA] = np.meshgrid(wavelength_range,analyte_index_range)

        plt.figure(figsize=(8,2), dpi=300)
        plt.subplot(1,2,1)
        plt.pcolormesh(LMC/1e-9,NAC,10*np.log10(abs(T_comsol[klc,:,:])))
        plt.xlim([1300,1900])
        plt.colorbar()
        plt.title('COMSOL, L = ' + str(float('%.3g' % (L_select/1e-6))) + 'um')
        plt.subplot(1,2,2)
        plt.pcolor(LM/1e-9,NA,10*np.log10(np.abs(T_CMT[:,kl,:])))
        plt.title('CMT, L = ' + str(float('%.3g' % (L_select/1e-6))) + 'um')
        plt.colorbar()
        plt.show()

def conventional_approach(L_propagate,index_to_plot,wavelength_range,analyte_index_range,neff_1,neff_2):
    # separate the imaginary parts into highest-loss and lowest-loss modes

    neff_min = np.zeros(neff_1.shape) # high loss mode
    neff_max = np.zeros(neff_1.shape) # low loss mode
    T_conventional = np.zeros(neff_1.shape)
    
    ka_index = np.zeros(index_to_plot.shape) # low loss mode

    for ka in range(0,len(analyte_index_range)):
        for kl in range(0,len(wavelength_range)):
            neff_min[ka,kl] =np.min(np.imag([neff_2[ka,kl],neff_1[ka,kl]]))
            neff_max[ka,kl] =np.max(np.imag([neff_2[ka,kl],neff_1[ka,kl]]))
        
    for ka in range(0,len(analyte_index_range)):
        T_conventional[ka,:] = np.exp(-4*np.pi/wavelength_range*neff_min[ka,:]*L_propagate)
    
    for ka in range(0,len(index_to_plot)):
        na_select = index_to_plot[ka]
        ka_index[ka] = int(np.where(np.abs(na_select-analyte_index_range)==np.min(np.abs(na_select-analyte_index_range)))[0][0])
        
    plt.figure()
    for ka in ka_index:
        ka = int(ka)
        # plot low loss mode
        plt.plot(wavelength_range/1e-9,np.squeeze(neff_min[ka,:]),'-',label = str(float('%.3g' % analyte_index_range[ka])))

    plt.gca().set_prop_cycle(None)
    # plot high loss mode
    
    for ka in ka_index:
        ka = int(ka)
        plt.plot(wavelength_range/1e-9,np.squeeze(np.real(neff_max[ka,:])),'--')
    plt.ylim([0,0.1])
    plt.xlim([1300,1900])
    plt.ylabel('imag neff')
    plt.xlabel('wavelength [nm]')
    plt.legend()
    plt.title('solid curve for "conventional" resonance')
    plt.show()
    
    # plot transmittion using "conventional" approach - Eq. (14)
    plt.figure()
    for ka in ka_index:
        ka = int(ka)
        plt.plot(wavelength_range/1e-9,10*np.log10(np.squeeze(T_conventional[ka,:])),'-',label = str(float('%.3g' % analyte_index_range[ka])))
    plt.xlim([1300,1900])
    plt.xlabel('wavelength [nm]')
    plt.ylabel('T [dB]')
    plt.legend()
    plt.title('Fig. 8(a)')
    plt.show()
    
    
    return T_conventional

def find_FWHM(lm_temp,T_temp,plot_index):
    index_min = np.where(T_temp == np.min(T_temp))[0][0] # index where the minimum is
    lm_min = lm_temp[index_min] # wavelength where the minimum is
    index_half_left = np.where(np.abs(T_temp[range(0,index_min)]-np.min(T_temp[range(0,index_min)])-3)==np.min(np.abs(np.abs(T_temp[range(0,index_min)]-np.min(T_temp[range(0,index_min)])-3)))) # half width to the left
    lm_right = lm_temp[range(index_min,len(lm_temp))] # define wavelength to the right of minimum
    T_right = T_temp[range(index_min,len(lm_temp))] # define transmission to the right of minimum 
    index_half_right = np.where(np.abs(T_right-np.min(T_right)-3)==np.min(np.abs(T_right-np.min(T_right)-3))) # half-width to the right 
    FWHM = (lm_right[index_half_right]-lm_temp[index_half_left])

    if plot_index==1:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(lm_temp/1e-9,T_temp)
        plt.title('FWHM is ' + str(float('%.3g' % (FWHM[0]/1e-9))) + 'nm' )
        plt.plot(lm_temp[index_min]/1e-9,T_temp[index_min],'ko')
        plt.subplot(2,1,2)
        plt.plot(lm_temp/1e-9,T_temp,'-k',label='zoomed in to FWHM')
        plt.plot(lm_temp[index_min]/1e-9,T_temp[index_min],'ko')
        plt.plot([lm_temp[0]/1e-9,lm_temp[-1]/1e-9],[T_temp[index_min]+3,T_temp[index_min]+3],'--k')
        plt.ylim([T_temp[index_min]-1,T_temp[index_min]+4])
        plt.xlim([lm_temp[index_min]/1e-9-FWHM/1e-9*2.0/3,lm_temp[index_min]/1e-9+FWHM/1e-9*2.0/3])
        plt.legend()
        plt.xlabel('wavelength [nm]')
        plt.show()
    
    return lm_min, FWHM, index_min

def plot_FOM(wavelength_range,analyte_index_range,Transmission_total,na_min,na_max):
    # obtain FWHM and lambdaR for conventional case
    FWHM = np.zeros(analyte_index_range.size)
    lm_min = np.zeros(analyte_index_range.size)

    for ka in range(0,len(analyte_index_range)):
        T_temp = 10*np.log10(np.abs(np.squeeze(Transmission_total)))[ka,:]
        lm_min[ka], FWHM[ka], index_min_temp =  find_FWHM(wavelength_range,T_temp,0)

    # convert to nm
    FWHM= FWHM/1e-9

    index_max_temp = int(np.where(np.abs(na_min-analyte_index_range)==np.min(np.abs(na_min-analyte_index_range)))[0])
    index_min_temp = int(np.where(np.abs(na_max-analyte_index_range)==np.min(np.abs(na_max-analyte_index_range)))[0])

    # make sure indeces are in ascending order...
    index_min = min(index_min_temp,index_max_temp)
    index_max = max(index_min_temp,index_max_temp)

    # interpolate in the range of interest
    na_range_i = np.linspace(na_min,na_max,1000)
    z = np.polyfit(analyte_index_range[index_min:index_max], lm_min[index_min:index_max], 2)
    p = np.poly1d(z)
    sensitivity = np.real(derivative(na_range_i,p(na_range_i)/1e-9))
    # interpolate to allow for FOM calculation
    f = interp1d(analyte_index_range, FWHM)
    FOM = np.abs(1/(f(na_range_i)/sensitivity))
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(analyte_index_range,lm_min/1e-9,'g.')
    ax1.plot(na_range_i,p(na_range_i)/1e-9,'g-')
    ax2.plot(na_range_i,sensitivity,'b-')
    ax1.set_xlabel('analyte index')
    ax1.set_ylabel('resonant wavelength [nm]', color='g')
    ax2.set_ylabel('sensitivity [nm/RIU]', color='b')
    ax1.set_xlim([na_min,na_max])
    ax1.set_ylim([1550,1800])
    ax2.set_ylim([-3000,500])
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(na_range_i,FOM,'-g')
    ax2.plot(na_range_i,f(na_range_i),'--b')
    ax2.plot(analyte_index_range,FWHM,'ob')
    ax1.set_xlabel('wavelength [nm]')
    ax1.set_ylabel('FOM', color='g')
    ax2.set_ylabel('delta lambda', color='b')
    ax1.set_xlim([na_min,na_max])
#    ax2.set_ylim([0,200])
    plt.show()


def plot_dispersion_lossless(n_analyte,wavelength_range,analyte_index_range,neff_0,neff_s_ll,neff_1_ll,neff_2_ll,C):
    """
    plot the lossless dispersion for a chosen analyte index    
    ----------
    n_analyte:   ndarray

    Returns
    -------
    a plot of the dispersion at n_analyte
    """
    ka = int(np.where(np.abs(n_analyte-analyte_index_range)==np.min(np.abs(n_analyte-analyte_index_range)))[0])
    #plt.figure(figsize=(12, 15), dpi=200)
    plt.figure(figsize=(10, 4), dpi=100)
    plt.subplot(1,2,1)
    plt.plot(wavelength_range/1e-9,np.real(neff_0[ka,:]),'--',label='Core mode')
    plt.plot(wavelength_range/1e-9,np.real(neff_s_ll[ka,:]),'--',label='Plasmonic mode')
    plt.gca().set_prop_cycle(None)
    plt.plot(wavelength_range/1e-9,np.real(neff_1_ll[ka,:]),'-',label='Core mode')
    plt.plot(wavelength_range/1e-9,np.real(neff_2_ll[ka,:]),'-',label='Core mode')
    plt.title('na = ' + str(round(analyte_index_range[ka],2)) + ' lossless')
    plt.ylabel('neff')
    plt.xlim([1300,1900])
    plt.subplot(1,2,2)
    plt.plot(wavelength_range/1e-9,np.real(C[ka,:]/1e6),'-k')
    plt.xlim([1300,1900])
    plt.ylim([0,0.5])
    plt.xlim([1300,1900])
    plt.ylabel('kappa [um]^(-1)')
    plt.xlabel('wavelength [um]')

def plot_dispersion_lossy(n_analyte,wavelength_range,analyte_index_range,neff_0,neff_s,neff_1,neff_2,neff_1_CMT,neff_2_CMT):
    """
    plot the lossless dispersion for a chosen analyte index    
    ----------
    n_analyte:   ndarray

    Returns
    -------
    a plot of the dispersion at n_analyte
    """
    ka = int(np.where(np.abs(n_analyte-analyte_index_range)==np.min(np.abs(n_analyte-analyte_index_range)))[0])
    #plt.figure(figsize=(12, 15), dpi=200)
    plt.figure(figsize=(10, 4), dpi=100)
    plt.subplot(1,2,1)
    plt.plot(wavelength_range/1e-9,np.real(neff_0[ka,:]),'--',label='Core mode')
    plt.plot(wavelength_range/1e-9,np.real(neff_s[ka,:]),'--',label='Plasmonic mode')
    plt.gca().set_prop_cycle(None)
    plt.plot(wavelength_range/1e-9,np.real(neff_1[ka,:]),'-')
    plt.plot(wavelength_range/1e-9,np.real(neff_2[ka,:]),'-')
    plt.plot(wavelength_range/1e-9,np.real(neff_1_CMT[ka,:]),'xk',markersize=0.2)
    plt.plot(wavelength_range/1e-9,np.real(neff_2_CMT[ka,:]),'xk',markersize=0.2)
    plt.title('na = ' + str(round(analyte_index_range[ka],2)) + ' lossy')
    plt.ylabel('Re neff')
    plt.xlim([1300,1900])
    plt.subplot(1,2,2)
    plt.plot(wavelength_range/1e-9,np.imag(neff_0[ka,:]),'--',label='Core mode')
    plt.plot(wavelength_range/1e-9,np.imag(neff_s[ka,:]),'--',label='Plasmonic mode')
    plt.gca().set_prop_cycle(None)
    plt.plot(wavelength_range/1e-9,np.imag(neff_1[ka,:]),'-')
    plt.plot(wavelength_range/1e-9,np.imag(neff_2[ka,:]),'-')
    plt.plot(wavelength_range/1e-9,np.imag(neff_1_CMT[ka,:]),'xk',markersize=0.2)
    plt.plot(wavelength_range/1e-9,np.imag(neff_2_CMT[ka,:]),'xk',markersize=0.2)
    plt.ylabel('Im neff')
    plt.xlim([1300,1900])
    plt.ylabel('Im neff')
    plt.xlabel('wavelength [um]')

def allocate_index(analyte_index_range,wavelength_range):
    neff_allocated = np.zeros([len(analyte_index_range),len(wavelength_range)])+0*1j
    return neff_allocated

def allocate_field(analyte_index_range,wavelength_range,x):
    LM, X = np.meshgrid(wavelength_range,x) # create wavelength/space mesh
    field_allocated = np.zeros([len(analyte_index_range),LM.shape[0],LM.shape[1]])+0*1j
    return field_allocated