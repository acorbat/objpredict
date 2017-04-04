# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:08:18 2017

@author: Agus
"""
import numpy as np

import skimage.measure as meas

from mahotas.features import _zernike
from mahotas import center_of_mass

#%%

def zernike_moments(im, radius, degree=8, cm=None):
    """
    zvalues = zernike_moments(im, radius, degree=8, cm={center_of_mass(im)})
    Zernike moments through ``degree``. These are computed on a circle of
    radius ``radius`` centered around ``cm`` (or the center of mass of the
    image, if the ``cm`` argument is not used).
    Returns a vector of absolute Zernike moments through ``degree`` for the
    image ``im``.
    Parameters
    ----------
    im : 2-ndarray
        input image
    radius : integer
        the maximum radius for the Zernike polynomials, in pixels. Note that
        the area outside the circle (centered on center of mass) defined by
        this radius is ignored.
    degree : integer, optional
        Maximum degree to use (default: 8)
    cm : pair of floats, optional
        the centre of mass to use. By default, uses the image's centre of mass.
    Returns
    -------
    zvalues : dictionary
        Dictionary with complex Zernike moments
    References
    ----------
    Teague, MR. (1980). Image Analysis via the General Theory of Moments.  J.
    Opt. Soc. Am. 70(8):920-930.
    """
    zvalues = []
    if cm is None:
        c0,c1 = center_of_mass(im)
    else:
        c0,c1 = cm

    Y,X = np.mgrid[:im.shape[0],:im.shape[1]]
    P = im.ravel()

    def rescale(C, centre):
        Cn = C.astype(np.double)
        Cn -= centre
        Cn /= radius
        return Cn.ravel()
    Yn = rescale(Y, c0)
    Xn = rescale(X, c1)

    Dn = Xn**2
    Dn += Yn**2
    np.sqrt(Dn, Dn)
    np.maximum(Dn, 1e-9, out=Dn)
    k = (Dn <= 1.)
    k &= (P > 0)

    frac_center = np.array(P[k], np.double)
    frac_center = frac_center.ravel()
    frac_center /= frac_center.sum()
    Yn = Yn[k]
    Xn = Xn[k]
    Dn = Dn[k]
    An = np.empty(Yn.shape, np.complex)
    An.real = (Xn/Dn)
    An.imag = (Yn/Dn)

    Ans = [An**p for p in range(2,degree+2)]
    Ans.insert(0, An) # An**1
    Ans.insert(0, np.ones_like(An)) # An**0
    zvalues = {}
    for n in range(degree+1):
        for l in range(n+1):
            if (n-l)%2 == 0:
                z = _zernike.znl(Dn, Ans[l], frac_center, n, l)
                zvalues[n, l] = (z)
    return zvalues

def zernike_invariants(im, deg=3, dbg=None):
    """
    Ss = zernike_invariants(im, deg=3, dbg=None)
    Zernike invariants up to order "deg".
    Returns a vector of Zernike seudoinvariants up to order "deg" of the image "im".
    Image is rescaled with the factor (0.7/mu_00)^(1/2) so that its weight is always normalized to 0.7, and traslated to the center of mass.
    Parameters
    ----------
    im : 2-ndarray
        input image
    deg : integer, optional
        seudo invariants up to this order are computed (default: 3)
    dbg : list, optional
        List where string with calculation of moments is appended to verify if these are computed adecuately.
    Returns
    -------
    Ss : 1-ndarray of floats
        List of Zernike seudoinvariants computed.
    References
    ----------
    Teague, MR. (1980). Image Analysis via the General Theory of Moments.  J.
    Opt. Soc. Am. 70(8):920-930.
    """
    if dbg is None:
        dbg = []
    try:
        mu = meas.regionprops(im)[0]['moments_central']
    except:
        print(im.shape, im.dtype)
        raise
        
    rad = np.sqrt(mu[0,0]/0.7) # compresses space to make image scale invariant (amount of pixels is conserved)
    Zs = zernike_moments(im, rad, degree=deg)
    
    Ss = [abs(Zs[2, 0]), abs(Zs[2, 2])**2] # start with second order invariants
    dbg.append('A_20')
    dbg.append('A_22^2')
    
    # Add third order invariants
    for l in range(1, 3+1):
        if (3-l)%2==0:
            Ss.append(abs(Zs[3, l])**2)
            dbg.append('A_3'+str(l)+'^2')
    
    this_S = np.conj(Zs[3, 3]) * Zs[3, 1]**3
    this_S += np.conj(this_S)
    dbg.append('A_33*A_31^3')
    Ss.append(abs(this_S))
    
    this_S = np.conj(Zs[3, 1])**2 * Zs[2, 2]
    this_S += np.conj(this_S)
    dbg.append('A_31*^2 A_22')
    Ss.append(abs(this_S))
    
    # Higher order invariants can be added by induction
    for n in range(4, deg+1):
        to_add = n + 1
        for l in range(n+1, -1, -1):
            # Add the trivial invariants
            if (n-l)%2==0:
                if l>0:
                    this_S = abs(Zs[n, l])**2
                    this_str = 'A_'+str(n)+str(l)+'^2'
                    to_add -= 1
                elif l==0:
                    this_S = abs(Zs[n, l]) # Can check if this is real
                    this_str = 'A_'+str(n)+str(l)
                    to_add -= 1
                dbg.append(this_str)
                Ss.append(this_S)
            
        # Add the non-trivial combinations
        for l in range(n, 0, -2):
            
            possible_l2 = [x for x in range(l, 0, -1) if l%x==0]
            for l2 in possible_l2:
                if l2==n: # if it's Ann then it can't be
                    continue
                
                for n2 in range(2, n+1): # Choose lowest n2 that can pair with highest l2
                    if n2<l2:
                        continue
                    
                    if (n2-l2)%2==0:
                        break
                
                m = int(l/l2) # Find exponent of second term
                
                this_S = np.conj(Zs[n, l]) * Zs[n2, l2]**m
                this_S += np.conj(this_S)
                this_str = 'A_'+str(n)+str(l)+'* A_'+str(n2)+str(l2)+'^'+str(m)
                
                # last special case A_44 A_42^2
                if n==4 and l==4:
                    this_S = np.conj(Zs[4, 4]) * Zs[4, 2]**2
                    this_S += np.conj(this_S)
                    this_str = 'A_44* A_42^2'
                
                dbg.append(this_str)
                Ss.append(abs(this_S))
                break
    
    return np.asarray(Ss)


def hu_invariants(im):
    props = meas.regionprops(im)
    nu = props[0]['moments_normalized']
    hu = hu_from_normalized_moments(nu)
    return hu


def hu_from_normalized_moments(nu):
    hu = np.zeros((8, ), dtype=np.double)
    t0 = nu[3, 0] + nu[1, 2]
    t1 = nu[2, 1] + nu[0, 3]

    q0 = t0 * t0
    q1 = t1 * t1
    n4 = 4 * nu[1, 1]
    s = nu[2, 0] + nu[0, 2]
    d = nu[2, 0] - nu[0, 2]

    hu[0] = s
    hu[1] = d * d + n4 * nu[1, 1]
    hu[3] = q0 + q1
    hu[5] = d * (q0 - q1) + n4 * t0 * t1

    t0 *= q0 - 3 * q1
    t1 *= 3 * q0 - q1
    q0 = nu[3, 0]- 3 * nu[1, 2]
    q1 = 3 * nu[2, 1] - nu[0, 3]

    hu[2] = q0 * q0 + q1 * q1
    hu[4] = q0 * t0 + q1 * t1
    hu[6] = q1 * t0 - q0 * t1

    hu[7] = (nu[1, 1] * ( (nu[3, 0] + nu[1, 2]) ** 2 - (nu[0, 3] + nu[2, 1]) ** 2) -
             (nu[2, 0] - nu[0, 2]) * (nu[3, 0] + nu[1, 2] + nu[0, 3] + nu[2, 1])
             )
    return hu