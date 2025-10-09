from skimage.draw import disk, polygon
import numpy as np 

def create_shape(size):
    """Generates different shapes as binary masks."""
    shape_mask = np.zeros((size, size))
    center = (size // 2, size // 2)
    
    r = np.array([size * 0.2, size * 0.8, size * 0.8])
    c = np.array([size * 0.5, size * 0.2, size * 0.8])
    rr, cc = polygon(r, c)
    shape_mask[rr, cc] = .2

    object_support = np.where(shape_mask>.1, 1.0,0)
    
    rr, cc = disk(np.array(center)-np.array([0,int(0.1*size)]), size//8, shape=shape_mask.shape)
    shape_mask[rr, cc] = 0.5
    
    return 1-shape_mask, object_support

def generate_phase_profile(size, phase_type="gradient", phase_max=np.pi/10, period = 4):
    """Generates a phase profile of the given size."""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    if phase_type == "gradient":
        phase_profile = phase_max * x  # Linear gradient in the x-direction
    elif phase_type == "gaussian":
        phase_profile = phase_max * np.exp(-(x**2 + y**2) / 0.5)  # Gaussian phase profile
    elif phase_type == "random":
        phase_profile = np.random.uniform(0, phase_max, (size, size))  # Random phase
    elif phase_type == "sinusoidal":
        phase_profile = phase_max/2 * (np.sin(period * np.pi * x) + np.sin(period*np.pi*y))
    else:
        raise ValueError("Unsupported phase type")
    
    return phase_profile