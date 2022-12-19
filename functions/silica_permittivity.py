import numpy as np



def silica_permittivity(lm):
    """
    relative electric permittivity of silica
    --------
    lm          :        ndarray
                		 wavelength [m]
                         
    Returns
    -------
    eps_SiO2    :       complex ndarray
                        electric permittivity of silica

    Reference
    -------
    I. H. Malitson. 
    Interspecimen comparison of the refractive index of fused silica, 
    J. Opt. Soc. Am. 55, 1205-1208 (1965)
    """
    LM = lm*1e6   # wavelength [um]
    B1 = 0.6961663
    C1 = 0.0684043
    B2 = 0.4079426
    C2 = 0.1162414
    B3 = 0.8974794
    C3 = 9.896161
    eps_SiO2 = 1 + B1*LM**2/(LM**2-C1**2) + B2*LM**2/(LM**2-C2**2) + B3*LM**2./(LM**2-C3**2)
    return eps_SiO2
