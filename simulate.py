import numpy as np 
import matplotlib.pyplot as plt 

from joblib import Parallel, delayed
import scipy 
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display
from .pupil import combined_aberrations
from .objects import generate_phase_profile, create_shape
import h5py
import ipywidgets as widgets


def forward_fft(arr):
          
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))

def inverse_fft(arr):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr)))
    
    
    
class Sim:
    
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

        krow = np.linspace(-self.pad_factor*self.na/self.wavelength* 2 * np.pi, self.pad_factor*self.na/self.wavelength*2 * np.pi, self.n_pixels)
        kcol = np.linspace(-self.pad_factor*self.na/self.wavelength* 2 * np.pi, self.pad_factor*self.na/self.wavelength*2 * np.pi, self.n_pixels)
        
        self.kRow, self.kCol = np.meshgrid(krow, kcol, indexing = 'ij')
        
        row = np.linspace(-self.object_size//2, self.object_size//2, self.n_pixels)
        col = np.linspace(-self.object_size//2, self.object_size//2, self.n_pixels)
        self.Row, self.Col = np.meshgrid(row,col, indexing = 'ij')
        

    def make_pupil(self, coefficients):
    
        self.aberration_coefficients = coefficients
        
        pupil_aperture_width = self.na/self.wavelength*2*np.pi
        
        aperture_mask = ( abs(self.kRow)<= pupil_aperture_width ) * ( abs(self.kCol)<= pupil_aperture_width ) 
        
        phase = combined_aberrations(self.kRow, self.kCol, coefficients, self.wavelength)
        
        phase *= aperture_mask
        
        self.pupil = aperture_mask* np.exp(1j*phase)

        self.probe = inverse_fft(self.pupil) * self.intensity

        self.extract_pupil_roi(margin=0)
        roi = self.pupil_roi

        self.gt_pupil = self.pupil[roi[0]:roi[1],roi[2]:roi[3]]

        del phase
        del aperture_mask

    
    def extract_pupil_roi(self, margin=10):

        # half-width of aperture in frequency space
        k_cutoff = self.na / self.wavelength * 2 * np.pi

        # find indices that fall within cutoff
        mask_row = np.where(np.abs(self.kRow[:, 0]) <= k_cutoff)[0]
        mask_col = np.where(np.abs(self.kCol[0, :]) <= k_cutoff)[0]

        row_min, row_max = mask_row.min(), mask_row.max()
        col_min, col_max = mask_col.min(), mask_col.max()

        # add margins safely
        row_min = max(0, row_min - margin)
        row_max = min(self.pupil.shape[1], row_max + margin + 1)
        col_min = max(0, col_min - margin)
        col_max = min(self.pupil.shape[0], col_max + margin + 1)

        # pupil_roi = self.pupil[y_min:y_max, x_min:x_max]
        # kX_roi = self.kX[y_min:y_max, x_min:x_max]
        # kY_roi = self.kY[y_min:y_max, x_min:x_max]

        self.pupil_roi = (row_min, row_max, col_min, col_max)
    
        del mask_row
        del mask_col
        
        # return pupil_roi, kX_roi, kY_roi

    def make_object(self):
        
        object_amp, object_support = create_shape(self.Row.shape[0])
        object_pha = generate_phase_profile(self.Row.shape[0], phase_type= 'sinusoidal') * object_support
        self.complex_object = object_amp*np.exp(1j*object_pha)
        del object_amp, object_pha
        
    def generate_scan_positions(self, step_size):
        
        """Generates a scan grid over the object."""
        self.step_size = step_size
        
        self.scan_positions = [(row, col) for row in range(0, self.n_pixels , step_size) for col in range(0, self.n_pixels , step_size)]
        
    @staticmethod
    def simulate_one_pattern(complex_object, probe, scan_position):
    
        centre = np.array(complex_object.shape)//2
        shift = centre - np.array(scan_position)
        shfited_object = scipy.ndimage.shift(complex_object, shift, mode='constant', cval = 1.0)  
        exit_wave = shfited_object*probe
        
        intensity_pattern = np.abs(forward_fft(exit_wave))**2
        
        return intensity_pattern

    @classmethod
    def simulate_parallel_patterns(cls, complex_object, probe, scan_positions, n_jobs = -1):
        print("Simulating diffraction patterns ...")

        intensity_patterns = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(cls.simulate_one_pattern)(complex_object, probe, position) for position in scan_positions
        )

        intensity_patterns = np.array(intensity_patterns)
        print("Done.")
        return intensity_patterns
    
    def simulate_dataset(self, n_jobs = -1):
        """
        Returns a 4D dataset: (Nrow, Ncol, qrow, qcol).
        """
        # simulate exit waves
        self.diff_patterns = self.simulate_parallel_patterns(self.complex_object, self.probe, self.scan_positions, n_jobs=n_jobs)
        
        # reshape into 4D (Ny_scan, Nx_scan, qy, qx)
        Nrow = len(np.unique([p[0] for p in self.scan_positions]))
        Ncol = len(np.unique([p[1] for p in self.scan_positions]))
        qrow, qcol = self.diff_patterns[0].shape
        
        print("Making 4D dataset ...")
        self.dataset_4d = np.flip(self.diff_patterns.reshape(Nrow, Ncol, qrow, qcol).transpose(0,1,2,3), axis=(0,1))
        print("Done.")
    def make_coherent_images(self):

        roi = self.pupil_roi
        print(f"Making coherent imges ...")

        self.data_roi = self.dataset_4d[:,:,roi[0]:roi[1],roi[2]:roi[3]]
        self.krow_roi = self.kRow[roi[0]:roi[1],roi[2]:roi[3]]
        self.kcol_roi = self.kCol[roi[0]:roi[1],roi[2]:roi[3]]

        scan_row, scan_col, det_row, det_col = self.data_roi.shape
        
        self.images = self.data_roi.reshape(scan_row, scan_col, -1).transpose(2, 0, 1)
        
        self.ks = np.column_stack([self.krow_roi.ravel(), self.kcol_roi.ravel()])

        print(f"Done.")

    # ____________________________ Plotting ________________________________
    def plot_4d_dataset(self, pupil_roi=None):
        
        print(f"roi while plotting 4d: {self.pupil_roi}")

        if pupil_roi is not None:
            data_4d = self.dataset_4d[:,:,pupil_roi[0]:pupil_roi[1], pupil_roi[2]:pupil_roi[3]]
        else:
            data_4d = self.dataset_4d[:,:,self.pupil_roi[0]:self.pupil_roi[1], self.pupil_roi[2]:self.pupil_roi[3]]
        
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
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        coherent_image = data_4d[:,:,prow_slider.value, pcol_slider.value]
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
            coherent_image = data_4d[:,:,prow,pcol]
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

    
        
    def plot_pupil_array(self, vmin1 = None, vmax1=None, vmin2 = None, vmax2=None, figsize = (10,5)):

        image1 = np.abs(self.pupil)
        image2 = np.angle(self.pupil)

        fig, ax = plt.subplots(1,2, figsize=figsize)

        im1 = ax[0].imshow(image1, vmin = vmin1, vmax = vmax1)
        im2 = ax[1].imshow(image2, vmin = vmin2, vmax = vmax2)

        ax[0].set_title('pupil amplitude')
        ax[1].set_title('pupil phase')

        plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
        
    def plot_pupil(self, vmin1 = None, vmax1=None, vmin2 = None, vmax2=None, figsize = (10,5)):

        
        
        pupil = self.gt_pupil

        image1 = np.abs(pupil)
        image2 = np.angle(pupil)

        fig, ax = plt.subplots(1,2, figsize=figsize)

        im1 = ax[0].imshow(image1, vmin = vmin1, vmax = vmax1)
        im2 = ax[1].imshow(image2, vmin = vmin2, vmax = vmax2)

        ax[0].set_title('pupil amplitude')
        ax[1].set_title('pupil phase')

        plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    def plot_object(self, vmin1 = None, vmax1=None, vmin2 = None, vmax2=None, figsize = (10,5)):
        image1 = np.abs(self.complex_object)
        image2 = np.angle(self.complex_object)

        fig, ax = plt.subplots(1,2, figsize=figsize)

        im1 = ax[0].imshow(image1, vmin = vmin1, vmax = vmax1)
        im2 = ax[1].imshow(image2, vmin = vmin2, vmax = vmax2)

        ax[0].set_title('Object amplitude')
        ax[1].set_title('Object phase')

        plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
    def plot_object_ft(self, vmin1 = None, vmax1=None, vmin2 = None, vmax2=None, figsize = (10,5)):
        object_ft = forward_fft(self.complex_object)
        image1 = np.abs(object_ft)
        image2 = np.angle(object_ft)

        fig, ax = plt.subplots(1,2, figsize=figsize)

        im1 = ax[0].imshow(image1, vmin = vmin1, vmax = vmax1)
        im2 = ax[1].imshow(image2, vmin = vmin2, vmax = vmax2)

        ax[0].set_title('object FT amplitude')
        ax[1].set_title('object FT phase')

        plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
        
    def plot_probe(self, vmin1 = None, vmax1=None, vmin2 = None, vmax2=None, figsize = (10,5)):
        image1 = np.abs(self.probe)
        image2 = np.angle(self.probe)

        fig, ax = plt.subplots(1,2, figsize=figsize)

        im1 = ax[0].imshow(image1, vmin = vmin1, vmax = vmax1)
        im2 = ax[1].imshow(image2, vmin = vmin2, vmax = vmax2)

        ax[0].set_title('Probe amplitude')
        ax[1].set_title('Probe phase')

        plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    
    
    def save_simulation(self, file_path):

        print("Saving simulation ...")
        
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
            amp = np.abs(self.gt_pupil)
            pha = np.angle(self.gt_pupil)
            ground_truth.create_dataset("Pupil_amplitude", data=amp, compression="gzip")
            ground_truth.create_dataset("Pupil_phase", data=pha, compression="gzip")
            
            # Save probe function
            amp = np.abs(self.probe)
            pha = np.angle(self.probe)
            ground_truth.create_dataset("Probe_amplitude", data=amp, compression="gzip")
            ground_truth.create_dataset("Probe_phase", data=pha, compression="gzip")
            
            
            simulated_data = hf.create_group("Simulated_Data")

            simulated_data.create_dataset("Data_4d", data = self.data_roi, compression="gzip")

            simulated_data.create_dataset("coherent_images", data = self.images, compression='gzip')
            #simulated_data.create_dataset("diffraction_patterns", data = self.diff_patterns, compression='gzip')
            
            simulated_data.create_dataset("ks", data = self.ks, compression='gzip')

        print(f"Simulation saved to {file_path}")

    @classmethod
    def load_simulation(cls, file_path):
        instance = cls.__new__(cls)
        with h5py.File(file_path, 'r') as hf:
            # Load simulation parameters
            simulation_params = hf["Simulation_Params"]
            instance.na = simulation_params.attrs["numerical_aperture"]
            instance.wavelength = simulation_params.attrs["wavelength"]
            instance.intensity = simulation_params.attrs["beam_intensity"]
            instance.object_size = simulation_params.attrs["object_size"]
            instance.pad_factor = simulation_params.attrs["padding_factor"]
            instance.step_size = simulation_params.attrs["step_size"]
            instance.real_space_psize = simulation_params.attrs["real_space_pixel_size_nm"]
            
            # Load aberration coefficients
            aberration_coefficients = hf["Aberration_Coefficients"]
            instance.aberration_coefficients = dict(aberration_coefficients.attrs)
            
            # Load ground truth data
            ground_truth = hf["Ground_Truth"]
            
            # Reconstruct complex object
            obj_amp = ground_truth["Object_amplitude"][:]
            obj_pha = ground_truth["Object_phase"][:]
            instance.complex_object = obj_amp * np.exp(1j * obj_pha)
            
            # Reconstruct pupil
            pupil_amp = ground_truth["Pupil_amplitude"][:]
            pupil_pha = ground_truth["Pupil_phase"][:]
            instance.gt_pupil = pupil_amp * np.exp(1j * pupil_pha)
            
            # Reconstruct probe
            probe_amp = ground_truth["Probe_amplitude"][:]
            probe_pha = ground_truth["Probe_phase"][:]
            cls.probe = probe_amp * np.exp(1j * probe_pha)
            
            # Load simulated data
            simulated_data = hf["Simulated_Data"]
            instance.data_roi = simulated_data["Data_4d"][:]
            #instance.krow_roi = simulated_data["kx"][:]
            #instance.kcol_roi = simulated_data["ky"][:]
            instance.images = simulated_data["coherent_images"][:]
            #instance.diff_patterns = simulated_data["diffraction_patterns"][:]
            instance.ks = simulated_data["ks"][:]
        
        return instance
        
class Sim_complex(Sim):
    
    @staticmethod
    def simulate_one_pattern(complex_object, probe, scan_position):
    
        centre = np.array(complex_object.shape)//2
        shift = centre - np.array(scan_position)
        shfited_object = scipy.ndimage.shift(complex_object, shift, mode='constant', cval = 1.0)  #, shift[::-1]
        exit_wave = shfited_object*probe
        
        complex_pattern = forward_fft(exit_wave)
        
        return complex_pattern
        
    def plot_complex_images(self, vmin1 = None, vmax1 = None, vmin2=-np.pi, vmax2=np.pi, cmap1 = 'viridis', cmap2 = 'viridis'):
        """Displays a list of coherent images and allows scrolling through them via a slider."""
        
    
        num_images = self.images.shape[0]  # Number of images in the list
        
        # Create a slider for selecting the image index
        img_slider = widgets.IntSlider(min=0, max= num_images - 1, value=0, description="Image")
    
        # Create figure & axis once
        fig, axes = plt.subplots(1,2, figsize=(10, 5))
        
        # Initial image
        im1 = axes[0].imshow(np.abs(self.images[0]), vmin = vmin1, vmax = vmax1, cmap=cmap1)    
        axes[0].set_title(f"Image 1 {0}/{num_images - 1}")
        #axes[0].axis('off')  # Hide axes
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot the second image
        im2 = axes[1].imshow(np.angle(self.images[0]), vmin = vmin2, vmax = vmax2, cmap=cmap2)
        axes[1].set_title(f"Image 1 {0}/{num_images - 1}")
        #axes[1].axis('off')  # Hide axes
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        def update_image(img_idx):
            """Updates the displayed image when the slider is moved."""
            img1 = np.abs(self.images[img_idx])
    
            img2 = np.angle(self.images[img_idx])
            
            im1.set_data(img1)  # Update image data
            im1.set_clim(vmin1, vmax1)
    
            im2.set_data(img2)  # Update image data
            im2.set_clim(vmin2, vmax2)
            
            axes[0].set_title(f"Image 1 {img_idx}/{num_images - 1}")  # Update title
            axes[1].set_title(f"Image 2 {img_idx}/{num_images - 1}")  # Update title
            fig.canvas.draw_idle()  # Efficient redraw
    
        # Create interactive slider
        interactive_plot = widgets.interactive(update_image, img_idx=img_slider)
    
        display(interactive_plot)  # Show slider
        #display(fig)  # Display the figure
    def plot_4d_dataset(self, pupil_roi=None):
        
        print(f"roi while plotting 4d: {self.pupil_roi}")

        if pupil_roi is not None:
            data_4d = np.abs(self.dataset_4d[:,:,pupil_roi[0]:pupil_roi[1], pupil_roi[2]:pupil_roi[3]])**2
        else:
            data_4d = np.abs(self.dataset_4d[:,:,self.pupil_roi[0]:self.pupil_roi[1], self.pupil_roi[2]:self.pupil_roi[3]])**2
        
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
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        coherent_image = data_4d[:,:,prow_slider.value, pcol_slider.value]
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
            coherent_image = data_4d[:,:,prow,pcol]
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
    sim.extract_pupil_roi(margin=20)
    
    # Ptycho
    sim.generate_scan_positions(step_size=step_size)
    sim.simulate_dataset()
    
    # Save results
    fpath = f"simulation_output_{time_str}.h5"
    
    sim.save_simulation(fpath)
    
    print(f"Simulation complete. Data saved to {fpath}")