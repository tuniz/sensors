import numpy as np


def derivative(xx,yy):
    """
    numerical derivative of a function, y = f(x) --> f'(x) = dy/dx
    ----------
    xx,yy	:   	ndarray
			        real or complex
    Returns
    -------
    dyy_dxx     :   	complex ndarray

    """
    yy = yy+0*1j
    dxx = (xx[2]-xx[0])
    dyy_dxx_r =  np.zeros(xx.size-2)
    dyy_dxx_i =  np.zeros(xx.size-2)

    for k in range(1,xx.size-1):
        dyy_dxx_r[k-1] = np.real(yy[k+1]-yy[k-1])/dxx
        dyy_dxx_i[k-1] = np.imag(yy[k+1]-yy[k-1])/dxx

    yy0_r = np.real(4*yy[1]-yy[2]-3*yy[0])/(dxx)
    yyf_r = np.real(4*yy[-1-1]-yy[-1-2]-3*yy[-1])/(-dxx)
    dyy_dxx_r = np.hstack([yy0_r, dyy_dxx_r, yyf_r])

    yy0_i = np.imag(4*yy[1]-yy[2]-3*yy[0])/(dxx)
    yyf_i = np.imag(4*yy[-1-1]-yy[-1-2]-3*yy[-1])/(-dxx)
    dyy_dxx_i = np.hstack([yy0_i, dyy_dxx_i, yyf_i])

    dyy_dxx = dyy_dxx_r+1j*dyy_dxx_i 
    return dyy_dxx