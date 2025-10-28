import numpy as np 

def defocus_aberration(kX, kY, delta_z, wavelength):
    
    k0 = 2*np.pi/wavelength
    phase = (- delta_z / (2*k0)) * (kX**2+kY**2)
    
    optical_path_diff = (wavelength/(2*np.pi)) * phase
    
    return phase

def spherical_aberration(kx, ky, spherical_coeff):

    return spherical_coeff * (kx**2 + ky**2)**2

def coma_aberration(kx, ky, coma_coeff):

    return coma_coeff * ky * (kx**2 + ky**2)

def astigmatism_aberration(kx, ky, astigmatism_coeff):

    return astigmatism_coeff * ( kx * ky )


def combined_aberrations(kx, ky, coefficients, wavelength):
    """
    Combine multiple aberrations.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - coefficients: Dictionary of aberration coefficients.
    
    Returns:
    - Total phase error (in radians).
    """
    phase_error = 0.0
    phase_error += defocus_aberration(kx, ky, coefficients['defocus'], wavelength)
    phase_error += spherical_aberration(kx, ky, coefficients['spherical'])
    phase_error += coma_aberration(kx, ky, coefficients['coma'])
    phase_error += astigmatism_aberration(kx, ky, coefficients['astigmatism'])
    return phase_error




def make_pupil(kX, kY, na, wavelength, coefficients):
    
    pupil_aperture_width = na/wavelength*2*np.pi
    
    aperture_mask = ( abs(kX)<= pupil_aperture_width ) * ( abs(kY)<= pupil_aperture_width ) 
    
    phase = combined_aberrations(kX, kY, coefficients, wavelength)
    
    phase *= aperture_mask
    
    return aperture_mask* np.exp(1j*phase)

