import numpy as np



def gold_permittivity(lm):
    """
    relative electric permittivity of gold
    --------
    lm          :       ndarray
                        wavelength [m]
                         
    Returns
    -------
    er_drude    :       complex ndarray
                        permittivity of gold

    Reference
    -------
    A. D. Rakić, A. B. Djurišic, J. M. Elazar, and M. L. Majewski. 
    Optical properties of metallic films for vertical-cavity optoelectronic devices, 
    Appl. Opt. 37, 5271-5283 (1998)
    """
    w = 1239.8/(lm/1e-9) # convert from wavelength to eV
    wp = 9.03
    f0 = 0.760
    Op = np.sqrt(f0)*wp
    G0 = 0.053
    f1 = 0.024
    G1 = 0.241
    w1 = 0.415
    f2 = 0.010
    G2 = 0.345
    w2 = 0.830
    f3 = 0.071
    G3 = 0.870
    w3 = 2.969
    f4 = 0.601
    G4 = 2.494
    w4 = 4.304
    f5 = 4.384
    G5 = 2.214
    w5 = 13.32
    er1 = 1-Op**2/(w*(w-1j*G0))
    er2 = f1*wp**2/((w1**2-w**2)+1j*w*G1) + f2*wp**2/((w2**2-w**2)+1j*w*G2) + f3*wp**2/((w3**2-w**2)+1j*w*G3) + f4*wp**2/((w4**2-w**2)+1j*w*G4) + f5*wp**2/((w5**2-w**2)+1j*w*G5)    
    er_drude = np.conj(er1+er2)
    return er_drude
