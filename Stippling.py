import numpy as np
import sys
import os
from PIL import Image
import colorsys
from scipy.interpolate import RegularGridInterpolator

# Adjust sys.path to import Pixels from ../pixels/Pixels.py
current_dir = os.path.dirname(os.path.abspath(__file__))
pixels_path = os.path.abspath(os.path.join(current_dir, '..', 'pixels'))
if pixels_path not in sys.path:
    sys.path.append(pixels_path)

try:
    from Pixels import Screen
except ImportError:
    print(f"Error: Could not import 'Screen' from {pixels_path}")
    raise

class BaseStipple:
    """
    A class to handle color stippling process using image data and Screen object.
    """

    def __init__(self, image_path=None, width=None, height=None, nx=None, ny=None, pixel_width=1.0, pixel_height=1.0):
        """
        Initialize BaseStipple with an image and Screen dimensions.
        """
        self.image_data = None
        self.width = None
        self.height = None

        if image_path:
            self.input_image(image_path)
            h, w, _ = self.image_data.shape
            self.width = float(w)
            self.height = float(h)
        else:
            if width is None or height is None:
                raise ValueError("If image_path is not provided, width and height must be specified.")
            self.width = float(width)
            self.height = float(height)

        # If nx, ny not provided, use default or calculate from width/height and pixel size
        if nx is None:
            nx = int(self.width / pixel_width)
        if ny is None:
            ny = int(self.height / pixel_height)
            
        self.screen = Screen(nx, ny, pixel_width, pixel_height)
        self.color_map = {}
        
        # Interpolators
        self.r_interp = None
        self.g_interp = None
        self.b_interp = None
        
        # HSV/Density Interpolators (per color key)
        self.hsv_data = {} # {color_key: {'v': v_data, 's': s_data}}
        self.hsv_interpolators = {} # {color_key: {'v': interp, 's': interp}}
        self.density_interpolators = {} # {color_key: density_interp}
        self.dome_library = None

        if self.image_data is not None:
            self.create_rgb_interpolators()

    def input_image(self, image_path):
        """
        Load a color image and store it as a numpy array scaled to [0, 1].
        """
        img = Image.open(image_path).convert('RGB')
        # Use numpy to store image data, transpose if necessary to match coordinate system (width x height)
        # Standard PIL image is (height, width, 3)
        self.image_data = np.array(img).astype(float) / 255.0

    def create_rgb_interpolators(self):
        """
        Create interpolation functions for R, G, and B values from the input image.
        Uses RegularGridInterpolator for efficiency.
        """
        if self.image_data is None:
            return

        h, w, _ = self.image_data.shape
        # Coordinates for the grid (normalized to width and height of the screen/image area)
        # Note: image index 0 is top, we might need to flip or adjust if screen Y starts from bottom
        # Assuming screen (0,0) is bottom-left and image (0,0) is top-left
        # We'll use the image dimensions for the interpolator grid
        x = np.linspace(0, self.width, w)
        y = np.linspace(0, self.height, h)
        
        # RegularGridInterpolator expects (y, x) if data is (h, w)
        # We flip the image along Y axis so y=0 is at the bottom
        flipped_data = np.flipud(self.image_data)
        
        self.r_interp = RegularGridInterpolator((y, x), flipped_data[:, :, 0], bounds_error=False, fill_value=0)
        self.g_interp = RegularGridInterpolator((y, x), flipped_data[:, :, 1], bounds_error=False, fill_value=0)
        self.b_interp = RegularGridInterpolator((y, x), flipped_data[:, :, 2], bounds_error=False, fill_value=0)

    def set_color_map(self, color_map, type="rgb"):
        """
        Set the color_map attribute.
        color_map: dict mapping string key to RGB tuple (values in [0, 255] or [0, 1])
        """
        # Ensure values are normalized if they look like 0-255
        processed_map = {}
        if type == "hsv":
            print("target colors are given in HSV system")
            for k, v in color_map.items():
                # if any(val > 1.0 for val in v):
                #     processed_map[k] = [tuple(val / 255.0 for val in v[1:]), ]
                # else:
                #     processed_map[k] = tuple(v[1:])
                processed_map[k] = tuple(v)
        elif type == "rgb":
            print("target colors are given in RGB system")
            for k, v in color_map.items():
                rgb_ = np.copy(v)
                if all(val <= 1.0 for val in v):
                    for val in v:
                        val *= 255.0
                h, s, v = colorsys.rgb_to_hsv(*rgb_)
                processed_map[k] = tuple([h,s,v])
        self.color_map = processed_map

    def create_hsv_interpolators(self, color_key, tolerance = 0.02):
        """
        Create interpolation functions for 'value' and 'saturation' with respect to a specific color in the color_map.
        """
        if self.image_data is None:
            return
        if color_key not in self.color_map:
            raise ValueError(f"Color key '{color_key}' not in color_map.")

        target_ = self.color_map[color_key]
        h, w, _ = self.image_data.shape
        
        # Calculate Value (V) and Saturation (S) relative to target color
        # This implementation depends on how "with respect to a specific color" is defined.
        # Usually, for stippling, we want to know how much of 'target_rgb' is present.
        # A simple way is to calculate a similarity or project onto the color vector.
        # But the roadmap specifically asks for "value" and "saturation".
        
        # We'll convert the whole image to HSV relative to the target color? 
        # Or just use standard HSV but maybe weighted by proximity to target RGB?
        # Let's assume it means the HSV components of the image data.
        
        v_data = np.zeros((h, w))
        s_data = np.zeros((h, w))
        
        # Convert RGB to HSV
        # colorsys.rgb_to_hsv works on single pixels
        for i in range(h):
            for j in range(w):
                r, g, b = self.image_data[i, j]
                h_val, s_val, v_val = colorsys.rgb_to_hsv(r, g, b)
                diff = abs(h_val - target_[0])
                if min (diff, 1.0 - diff) <= tolerance:
                    v_data[i, j] = v_val
                    s_data[i, j] = s_val

        # Store raw HSV data
        self.hsv_data[color_key] = {'v': v_data, 's': s_data}

        # Create interpolators
        x = np.linspace(0, self.width, w)
        y = np.linspace(0, self.height, h)
        flipped_v = np.flipud(v_data)
        flipped_s = np.flipud(s_data)
        
        self.hsv_interpolators[color_key] = {
            'v': RegularGridInterpolator((y, x), flipped_v, bounds_error=False, fill_value=0),
            's': RegularGridInterpolator((y, x), flipped_s, bounds_error=False, fill_value=0)
        }

    def create_density_interpolators(self, k: float = 0.896):
        """
        Create interpolation functions for color density using raw HSV data.
        """
        if self.image_data is None:
            return

        h, w, _ = self.image_data.shape
        x = np.linspace(0, self.width, w)
        y = np.linspace(0, self.height, h)

        for color_key, data in self.hsv_data.items():
            v_data = np.array(data['v'])
            s_data = np.array(data['s'])

            # Compute density array: 1. - v * (1. - k * s)
            density_data = 1. - np.multiply(v_data, 1.0-k*s_data)
            
            # Ensure consistency with coordinate system (flipud)
            flipped_density = np.flipud(density_data)
            
            # Create a new RegularGridInterpolator instance
            self.density_interpolators[color_key] = RegularGridInterpolator(
                (y, x), flipped_density, bounds_error=False, fill_value=0
            )

    def set_dome_library(self, dome_library):
        """
        Set the dome_library attribute.
        """
        self.dome_library = dome_library

    def get_dome_library(self):
        """
        Get the dome_library attribute.
        """
        return self.dome_library
