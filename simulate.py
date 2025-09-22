import numpy as np 
import matplotlib.pyplot as plt 

from joblib import Parallel, delayed
import scipy 
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display
from pupil import combined_aberrations
from objects import generate_phase_profile, create_shape
import h5py


def forward_fft(arr):
        
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))

def inverse_fft(arr):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr)))
    
    
    
class simulate:
    
    def __init__(self, na, wavelength, intensity = 1e6):
        
        self.na = na
        self.wavelength = wavelength  
        self.intensity = intensity   
    
    @property
    def object_size(self):
        
        return self._object_size
    
    @object_size.setter
    def object_size(self, object_size):
        self._object_size = object_size
        
    @property
    def pad_factor(self):
        
        return self._pad_factor
    
    @pad_factor.setter
    def pad_factor(self, pad_factor=16):
        self._pad_factor = pad_factor
            
        
        
    def make_grids(self):
        
        fmax = 2*self.pad_factor*self.na/self.wavelength * 2 * np.pi
        
        delta_fx = 2*np.pi/self.object_size
        
        self.n_pixels = (round(fmax/delta_fx)//2)*2
        
        self.real_space_psize = self.object_size/self.n_pixels # m 

        kx = np.linspace(-self.pad_factor*self.na/self.wavelength* 2 * np.pi, self.pad_factor*self.na/self.wavelength*2 * np.pi, self.n_pixels)
        ky = np.linspace(-self.pad_factor*self.na/self.wavelength* 2 * np.pi, self.pad_factor*self.na/self.wavelength*2 * np.pi, self.n_pixels)
        
        self.kX,self.kY = np.meshgrid(kx, ky)
        
        x = np.linspace(-self.object_size//2, self.object_size//2, self.n_pixels)
        y = np.linspace(-self.object_size//2, self.object_size//2, self.n_pixels)
        self.X, self.Y = np.meshgrid(x,y)
        

    def make_pupil(self, coefficients):
    
        self.aberration_coefficients = coefficients
        
        pupil_aperture_width = self.na/self.wavelength*2*np.pi
        
        aperture_mask = ( abs(self.kX)<= pupil_aperture_width ) * ( abs(self.kY)<= pupil_aperture_width ) 
        
        phase = combined_aberrations(self.kX, self.kY, coefficients, self.wavelength)
        
        phase *= aperture_mask
        
        self.pupil = aperture_mask* np.exp(1j*phase)

        self.probe = inverse_fft(self.pupil) * self.intensity
    
    def extract_pupil_roi(self, margin=10):

        # half-width of aperture in frequency space
        k_cutoff = self.na / self.wavelength * 2 * np.pi

        # find indices that fall within cutoff
        mask_x = np.where(np.abs(self.kX[0, :]) <= k_cutoff)[0]
        mask_y = np.where(np.abs(self.kY[:, 0]) <= k_cutoff)[0]

        x_min, x_max = mask_x.min(), mask_x.max()
        y_min, y_max = mask_y.min(), mask_y.max()

        # add margins safely
        x_min = max(0, x_min - margin)
        x_max = min(self.pupil.shape[1], x_max + margin + 1)
        y_min = max(0, y_min - margin)
        y_max = min(self.pupil.shape[0], y_max + margin + 1)

        # pupil_roi = self.pupil[y_min:y_max, x_min:x_max]
        # kX_roi = self.kX[y_min:y_max, x_min:x_max]
        # kY_roi = self.kY[y_min:y_max, x_min:x_max]

        self.pupil_roi = (y_min, y_max, x_min, x_max)
        
        # return pupil_roi, kX_roi, kY_roi

    def make_object(self):
        
        object_amp, object_support = create_shape(self.X.shape[0])
        object_pha = generate_phase_profile(self.X.shape[0], phase_type= 'sinusoidal') * object_support
        self.complex_object = object_amp*np.exp(1j*object_pha)
        
    def generate_scan_positions(self, step_size):
        
        """Generates a scan grid over the object."""
        self.step_size = step_size
        
        self.scan_positions = [(x, y) for x in range(0, self.n_pixels , step_size) for y in range(0, self.n_pixels , step_size)]
        
    @staticmethod
    def simulate_one_pattern(complex_object, probe, scan_position):
    
        centre = np.array(complex_object.shape)//2
        shift = centre - np.array(scan_position)
        shfited_object = scipy.ndimage.shift(complex_object, shift[::-1], mode='constant', cval = 1.0)  
        exit_wave = shfited_object*probe

        fft_exit_wave = forward_fft(exit_wave)
        diff_pattern = np.abs(fft_exit_wave)**2
        
        return diff_pattern

    @classmethod
    def simulate_parallel_patterns(cls, complex_object, probe, scan_positions):
        
        patterns = Parallel(n_jobs=-1)(
            delayed(cls.simulate_one_pattern)(complex_object, probe, position) for position in scan_positions
        )
        return np.array(patterns)
    
    def simulate_dataset(self):
        """
        Returns a 4D dataset: (Ny, Nx, qy, qx).
        """
        # simulate exit waves
        patterns = self.simulate_parallel_patterns(self.complex_object, self.probe, self.scan_positions)
        
        # reshape into 4D (Ny_scan, Nx_scan, qy, qx)
        Nx = len(np.unique([p[0] for p in self.scan_positions]))
        Ny = len(np.unique([p[1] for p in self.scan_positions]))
        qy, qx = patterns[0].shape
        
        self.diff_patterns = patterns

        self.dataset_4d = patterns.reshape(Ny, Nx, qy, qx).transpose(1,0,2,3)

    def plot_4d_dataset(data_4d, pupil_roi=None):

        if pupil_roi is not None:
            data_4d = data_4d[:,:,pupil_roi[0]:pupil_roi[1], pupil_roi[2]:pupil_roi[3]]
            
        # Get dataset dimensions
        coherent_shape = data_4d.shape[:2]  
        detector_shape = data_4d.shape[2:]  
            
        # Set slider limits
        pcol_slider = widgets.IntSlider(min=0, max= detector_shape[1] - 1, value=detector_shape[1]//2, description="px")
        prow_slider = widgets.IntSlider(min=0, max= detector_shape[0] - 1, value=detector_shape[0]//2, description="py")
        
        lcol_slider = widgets.IntSlider(min=0, max= coherent_shape[1] - 1, value=coherent_shape[1]//2, description="lx")
        lrow_slider = widgets.IntSlider(min=0, max= coherent_shape[0] - 1, value=coherent_shape[0]//2, description="ly")


        rectangle_size_det = 4 
        rectangle_size_coh = .5
        
        # Create the figure and axes **only once**
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))

        coherent_image = data_4d[:,:,prow_slider.value, pcol_slider.value].T
        detector_image = data_4d[lrow_slider.value, lcol_slider.value,:,:]

        
        im0 = axes[0].imshow(coherent_image, cmap='plasma')
        
        axes[0].set_title(f"Coherent Image (lx={lrow_slider.value}, ly={lcol_slider.value})")
        rect_coherent = Rectangle(( lcol_slider.value - rectangle_size_coh / 2, lrow_slider.value - rectangle_size_coh / 2), 
                                rectangle_size_coh, rectangle_size_coh, 
                                edgecolor='white', facecolor='white', lw=2)
        
        axes[0].add_patch(rect_coherent)
        plt.colorbar(im0, ax=axes[0], label="Intensity")

        im1 = axes[1].imshow(detector_image, cmap='viridis')
        axes[1].set_title(f"Detector Image (px={pcol_slider.value}, py={prow_slider.value})")
        rect_detector = Rectangle((pcol_slider.value - rectangle_size_det / 2, prow_slider.value - rectangle_size_det / 2), 
                                rectangle_size_det, rectangle_size_det, 
                                edgecolor='white', facecolor='white', lw=2)
        axes[1].add_patch(rect_detector)
        plt.colorbar(im1, ax=axes[1], label="Detector Intensity")

        plt.tight_layout()
        
        def update_plot(prow, pcol, lrow, lcol):
            """ Updates the plot based on slider values. """
            coherent_image = data_4d[:,:,prow,pcol].T
            detector_image = data_4d[lrow,lcol,:,:]

            im0.set_data(coherent_image)
            im1.set_data(detector_image)

            axes[0].set_title(f"Coherent Image from Pixel ({pcol}, {prow})")
            axes[1].set_title(f'Detector Image at Location ({lcol},{lrow})')
            
            # Update rectangles
            rect_coherent.set_xy((lcol - rectangle_size_coh / 2, lrow - rectangle_size_coh / 2))
            rect_detector.set_xy((pcol - rectangle_size_det / 2, prow - rectangle_size_det / 2))


            fig.canvas.draw_idle()
            
        # Create interactive widget
        interactive_plot = widgets.interactive(update_plot, prow=prow_slider, pcol=pcol_slider, lrow=lrow_slider, lcol=lcol_slider)
        
        # controls = widgets.VBox([prow_slider, pcol_slider, lrow_slider, lcol_slider])
        display(interactive_plot)
        plt.show()
        
    def save_simulation(self, file_path):
        
        
        meta_data = {
            "numerical_aperture": self.na,
            "wavelength": self.wavelength,
            "beam_intensity": self.intensity,
            "object_size": self.object_size,
            "padding_factor": self.pad_factor,
            "step_size": self.step_size, 
            "real_space_pixel_size_nm": self.real_space_psize
        }
        
        
        with h5py.File(file_path, 'w') as hf:
            
            simulation_params = hf.create_group("Simulation_Params")

            aberration_coefficients = hf.create_group("Aberration_Coefficients")
            
            for key, value in meta_data.items():
                simulation_params.attrs[key] = value
            
            for key, value in self.aberration_coefficients.items():
                aberration_coefficients.attrs[key] = value
                
            ground_truth = hf.create_group("Ground_Truth")
            
            # Save object images 
            amp = np.abs(self.complex_object)
            pha = np.angle(self.complex_object)
            
            
            ground_truth.create_dataset("Object_amplitude", data=amp, compression="gzip")
            ground_truth.create_dataset("Object_phase", data=pha, compression="gzip")
            
            # Save Pupil 
            amp = np.abs(self.pupil)
            pha = np.angle(self.pupil)
            ground_truth.create_dataset("Pupil_amplitude", data=amp, compression="gzip")
            ground_truth.create_dataset("Pupil_phase", data=pha, compression="gzip")
            
            # Save probe function
            amp = np.abs(self.probe)
            pha = np.angle(self.probe)
            ground_truth.create_dataset("Probe_amplitude", data=amp, compression="gzip")
            ground_truth.create_dataset("Probe_phase", data=pha, compression="gzip")
            
            
            simulated_data = hf.create_group("Simulated_Data")
            
            roi = self.pupil_roi
            simulated_data.create_dataset("Data_4d", data = self.dataset_4d[:,:,roi[0]:roi[1],roi[2]:roi[3]], compression="gzip")
            simulated_data.create_dataset("kx", data = self.kX[roi[0]:roi[1],roi[2]:roi[3]], compression="gzip")
            simulated_data.create_dataset("ky", data = self.kY[roi[0]:roi[1],roi[2]:roi[3]], compression="gzip")
        
if __name__ == "__main__":
    
    from time import strftime
    time_str = strftime("%Y-%m-%d_%H.%M")
    
    # Define simulation parameters
    na = 0.1
    wavelength = .7 * 1e-10 # m
    object_size = 1e-6  # m
    pad_factor = 16 # Multiplier
    step_size = 16 # Pixels
    
    coefficients = {'defocus': 100 * 1e-6, # m 
                'spherical': 0,
                'coma': 0, 
                'astigmatism': 0}
    
    # Create simulator
    sim = simulate(na=na, wavelength=wavelength, intensity=1e6)
    sim.object_size = object_size
    sim.pad_factor = pad_factor
    
    sim.make_grids()
    
    # Build pupil, probe and object
    sim.make_pupil(coefficients)
    sim.make_object()

    # Extract ROI; 
    # For efficient data saving
    sim.extract_pupil_roi(margin=10)
    
    # Ptycho
    sim.generate_scan_positions(step_size=step_size)
    sim.simulate_dataset()
    
    # Save results
    fpath = f"simulation_output_{time_str}.h5"
    
    sim.save_simulation(fpath)
    
    print(f"Simulation complete. Data saved to {fpath}")