import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import sys
import os
import pickle

# Add the path to dithering for Lloyd.py
current_dir = os.path.dirname(os.path.abspath(__file__))
dithering_path = os.path.abspath(os.path.join(current_dir, '..', 'dithering'))
if dithering_path not in sys.path:
    sys.path.append(dithering_path)

try:
    import Lloyd
except ImportError:
    pass # Will be handled if Lloyd is not available

sys.path.insert(1, '/home/jaeukk/30_Codes/python/Common/')
import drawings as dr

class dualmethod:
    """
    A decorator that allows a method to be called on both the class and an instance.
    When called on the class, the first argument is the class.
    When called on an instance, the first argument is the instance.
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, instance, owner):
        return lambda *args, **kwargs: self.func(instance if instance is not None else owner, *args, **kwargs)

class Pixel:
    def __init__(self, width: float, height: float, dimension: int = 2):
        self.dimension = dimension
        self.width = float(width)
        self.height = float(height)
        self.positions = np.empty((0, self.dimension), dtype=float)
        self.radii = np.empty(0, dtype=float)
        self.types = np.empty(0, dtype=int)
        
    def copy(self):
        """
        Returns a new Pixel instance that is a copy of the current one.
        """
        new_pixel = Pixel(self.width, self.height, self.dimension)
        new_pixel.positions = self.positions.copy()
        new_pixel.radii = self.radii.copy()
        new_pixel.types = self.types.copy()
        return new_pixel
        
    def input(self, positions, radii, particle_numbers):
        """
        Inputs particles into the pixel.
        positions: list or array of particle positions (in local coordinates)
        radii: list or array of particle radii
        particle_numbers: array of particle numbers per type, its sum should equal the total particle number
        """

        if len(self.positions) > 0 and len(self.types) != len(self.positions):
            raise ValueError(f"Sum of particle numbers ({len(self.types)}) does not match number of positions ({len(self.positions)}).")
        else:

            self.positions = np.array(positions, dtype=float)
            
            # Radii can be given as one per particle, or one per type
            radii_arr = np.array(radii, dtype=float)
            if len(radii_arr) == len(self.positions):
                self.radii = radii_arr
            elif len(radii_arr) == len(particle_numbers):
                self.radii = np.repeat(radii_arr, np.array(particle_numbers, dtype=int))
            elif len(radii_arr) == 1:
                self.radii = np.repeat(radii_arr, len(self.positions))
            else:
                self.radii = radii_arr # Let it pass, maybe handled externally
                
            types = []
            for type_idx, count in enumerate(particle_numbers):
                types.extend([type_idx] * int(count))
            self.types = np.array(types, dtype=int)
            

class Screen:
    def __init__(self, nx: int, ny: int, pixel_width: float, pixel_height: float):
        self.nx = nx
        self.ny = ny
        self.pixel_width = float(pixel_width)
        self.pixel_height = float(pixel_height)
        self.pixels = [[Pixel(self.pixel_width, self.pixel_height) for _ in range(ny)] for _ in range(nx)]
        
    def copy(self):
        """
        Returns a new Screen instance that is a deep copy of the current one.
        """
        new_screen = Screen(self.nx, self.ny, self.pixel_width, self.pixel_height)
        new_screen.pixels = [[self.pixels[i][j].copy() for j in range(self.ny)] for i in range(self.nx)]
        return new_screen
        
    def update_pixel(self, index, positions, radii, particle_numbers):
        """
        Updates a Pixel object at the given index.
        index: tuple (row, col)
        positions: list or array of particle positions (in local coordinates)
        radii: list or array of particle radii
        particle_numbers: array of particle numbers per type, its sum should equal the total particle number
        """
        row, col = index
        self.pixels[row][col].input(positions, radii, particle_numbers)
        
    def get_pixel_index(self, position):
        """
        Returns the array indices (i, j) of the Pixel containing the given 2D position.
        """
        x, y = position
        i = int(x // self.pixel_width)
        j = int(y // self.pixel_height)
        
        # Bound the indices to be within the screen
        i = max(0, min(i, self.nx - 1))
        j = max(0, min(j, self.ny - 1))
        return (i, j)
        
    def get_surrounding_indices(self, index):
        """
        Returns the array indices of the Pixel objects surrounding an input index.
        """
        i, j = index
        surrounding = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.nx and 0 <= nj < self.ny:
                    surrounding.append((ni, nj))
        return surrounding
        
    def get_pixel_positions(self):
        """
        Returns the positions of the lower left corners of the Pixels.
        Output shape: (nx, ny, 2)
        """
        positions = np.zeros((self.nx, self.ny, 2))
        for i in range(self.nx):
            for j in range(self.ny):
                positions[i, j] = [i * self.pixel_width, j * self.pixel_height]
        return positions
        
    def visualize(self, fig_handler, styles, indices=None, use_collection=False, radius_to_pixel=1.0):
        """
        Visualizes all or portions of Pixel objects using Plot2DPacking or EllipseCollection.
        fig_handler: [fig, ax] passed to Plot2DPacking
        styles: list of dictionaries specifying styles for each particle type
        indices: list of (i, j) tuples specifying which pixels to visualize. If None, all are visualized.
        use_collection: If True, uses EllipseCollection for efficiency.
        radius_to_pixel: length unit of the particle radius in pixels.
        """
        if indices is None:
            indices = [(i, j) for i in range(self.nx) for j in range(self.ny)]
            
        screen_basis = np.array([[self.nx * self.pixel_width, 0], 
                                 [0, self.ny * self.pixel_height]])
            
        if not use_collection:
            for (i, j) in indices:
                pixel = self.pixels[i][j]
                if len(pixel.positions) == 0:
                    continue
                
                # Calculate absolute positions
                offset = np.array([i * self.pixel_width, j * self.pixel_height])
                abs_positions = pixel.positions + offset
                
                unique_types = np.unique(pixel.types)
                for t in unique_types:
                    mask = (pixel.types == t)
                    pos = abs_positions[mask]
                    rad = pixel.radii[mask] * radius_to_pixel
                    style = styles[t] if t < len(styles) else {}
                    dr.Plot2DPacking(fig_handler, screen_basis, pos, rad, style)
        else:
            # Use EllipseCollection to visualize dots
            fig, ax = fig_handler
            
            # Group data by type for efficiency
            type_data = {}
            for (i, j) in indices:
                pixel = self.pixels[i][j]
                if len(pixel.positions) == 0:
                    continue
                
                offset = np.array([i * self.pixel_width, j * self.pixel_height])
                abs_pos = pixel.positions + offset
                
                unique_types = np.unique(pixel.types)
                for t in unique_types:
                    mask = (pixel.types == t)
                    if t not in type_data:
                        type_data[t] = {'pos': [], 'rad': []}
                    type_data[t]['pos'].append(abs_pos[mask])
                    type_data[t]['rad'].append(pixel.radii[mask])
            
            # Set limits and aspect to ensure correct scale calculation
            xmin, xmax = 0, self.nx * self.pixel_width
            ymin, ymax = 0, self.ny * self.pixel_height
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect(1)
            
            # Using EllipseCollection for exact sizing in data units, matching mpatches.Circle
            for t, data in type_data.items():
                final_pos = np.vstack(data['pos'])
                final_rad = np.concatenate(data['rad'])
                style = styles[t] if t < len(styles) else {}
                
                kwargs = {}
                if 'facecolor' in style: kwargs['facecolors'] = style['facecolor']
                if 'edgecolor' in style: kwargs['edgecolors'] = style['edgecolor']
                if 'linewidth' in style: kwargs['linewidths'] = style['linewidth']
                if 'alpha' in style: kwargs['alpha'] = style['alpha']
                
                diameters = final_rad * 2 * radius_to_pixel
                angles = np.zeros_like(diameters)
                collection = EllipseCollection(
                    diameters, diameters, angles, units='x', offsets=final_pos,
                    transOffset=ax.transData, **kwargs
                )
                ax.add_collection(collection)
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    def check_overlaps(self):
        """
        Checks whether there are any overlapping particles across the screen.
        Returns a list of tuples ((i, j), k) representing the pixel indices (i, j)
        and particle index k within that pixel for particles that overlap with others.
        """
        overlapping_particles = set()
        
        for i in range(self.nx):
            for j in range(self.ny):
                pixel = self.pixels[i][j]
                if len(pixel.positions) == 0:
                    continue
                
                offset_c = np.array([i * self.pixel_width, j * self.pixel_height])
                pos_c = pixel.positions + offset_c
                rad_c = pixel.radii
                
                # Check within same pixel
                for k in range(len(pos_c)):
                    for m in range(k + 1, len(pos_c)):
                        dist = np.linalg.norm(pos_c[k] - pos_c[m])
                        if dist < rad_c[k] + rad_c[m] + 1e-8:
                            overlapping_particles.add(((i, j), k))
                            overlapping_particles.add(((i, j), m))
                
                # Check with surrounding pixels
                for k in range(len(pos_c)):
                    local_x, local_y = pixel.positions[k]
                    r = rad_c[k]
                    
                    cross_left = local_x - r < 0
                    cross_right = local_x + r > self.pixel_width
                    cross_bottom = local_y - r < 0
                    cross_top = local_y + r > self.pixel_height
                    
                    if not (cross_left or cross_right or cross_bottom or cross_top):
                        continue
                        
                    neighbors_to_check = set()
                    if cross_left:
                        neighbors_to_check.add((i - 1, j))
                        if cross_bottom:
                            neighbors_to_check.add((i - 1, j - 1))
                        if cross_top:
                            neighbors_to_check.add((i - 1, j + 1))
                    if cross_right:
                        neighbors_to_check.add((i + 1, j))
                        if cross_bottom:
                            neighbors_to_check.add((i + 1, j - 1))
                        if cross_top:
                            neighbors_to_check.add((i + 1, j + 1))
                    if cross_bottom:
                        neighbors_to_check.add((i, j - 1))
                    if cross_top:
                        neighbors_to_check.add((i, j + 1))
                        
                    for (ni, nj) in neighbors_to_check:
                        if 0 <= ni < self.nx and 0 <= nj < self.ny:
                            surr_pixel = self.pixels[ni][nj]
                            if len(surr_pixel.positions) == 0:
                                continue
                            
                            offset_n = np.array([ni * self.pixel_width, nj * self.pixel_height])
                            pos_n = surr_pixel.positions + offset_n
                            rad_n = surr_pixel.radii
                            
                            for m in range(len(pos_n)):
                                dist = np.linalg.norm(pos_c[k] - pos_n[m])
                                if dist < r + rad_n[m] + 1e-8:
                                    overlapping_particles.add(((i, j), k))
                                    overlapping_particles.add(((ni, nj), m))
                                    
        return list(overlapping_particles)

    def write_to_file(self, savefilename: str, header: str, decoration: dict):
        """
        Writes all particles in the screen's pixels to a text file.
        savefilename: output file path.
        header: string to be written at the top.
        decoration: dictionary mapping integer particle types to strings.
        """
        with open(savefilename, "w") as f:
            if header:
                if not header.endswith('\n'):
                    f.write(header + '\n')
                else:
                    f.write(header)
            
            id_counter = 1
            for i in range(self.nx):
                for j in range(self.ny):
                    offset = np.array([i * self.pixel_width, j * self.pixel_height])
                    pixel = self.pixels[i][j]
                    for k in range(len(pixel.positions)):
                        pos = pixel.positions[k] + offset
                        species = decoration[pixel.types[k]]
                        f.write("{:d}\t{:.10e}\t{:0.10e}\t{:s}\n".format(id_counter, pos[0], pos[1], species))
                        id_counter += 1

    def save(self, filepath: str):
        """
        Saves the current Screen instance to a file using pickle.
        
        Saved data includes:
        - Grid dimensions (nx, ny)
        - Pixel dimensions (pixel_width, pixel_height)
        - Full nested list of Pixel objects, including their:
            - positions (numpy.ndarray)
            - radii (numpy.ndarray)
            - types (numpy.ndarray)
        
        filepath: path to the output file.
        """
        # Ensure parent directory exists
        parent_dir = os.path.dirname(filepath)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        try:
            with open(filepath, 'wb') as f:
                # Use HIGHEST_PROTOCOL for efficiency and better numpy support
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise IOError(f"Failed to save Screen to {filepath}: {e}")

    @dualmethod
    def load(self_or_cls, filepath: str):
        """
        Loads a Screen instance from a file using pickle.
        
        If called as a class method (e.g., Screen.load(path)), it returns a new instance.
        If called as an instance method (e.g., myscreen.load(path)), it updates 
        the instance's state with the loaded data and returns self.
        
        filepath: path to the pickle file.
        Returns: A Screen instance (either a new one or self).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Screen file not found: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            
            # Robustness: Check if loaded object is indeed a Screen
            # Determine the target class for validation
            cls = self_or_cls if isinstance(self_or_cls, type) else type(self_or_cls)
            
            if not isinstance(obj, cls) and obj.__class__.__name__ != cls.__name__:
                raise TypeError(f"Loaded object is of type {type(obj)}, expected {cls}")
            
            # Sanity check for essential attributes
            required_attrs = ['nx', 'ny', 'pixels', 'pixel_width', 'pixel_height']
            for attr in required_attrs:
                if not hasattr(obj, attr):
                    raise AttributeError(f"Loaded Screen object is missing attribute: {attr}")
            
            if isinstance(self_or_cls, type):
                # Class method call: return the new instance
                return obj
            else:
                # Instance method call: update self.__dict__
                self_or_cls.__dict__.update(obj.__dict__)
                return self_or_cls
            
        except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
            raise RuntimeError(f"Failed to unpickle Screen from {filepath}. "
                               f"This may be due to module path changes or file corruption: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading Screen from {filepath}: {e}")

def Relax(screen: Screen, alpha=0.2, iterations=1, indices=None):
    """
    Applies the ordinary Lloyd algorithm in each Pixel object.
    Uses the particles in the Pixels surrounding the target Pixel as padding.
    """
    # Create a copy of the screen pixels to avoid updating while reading neighbors
    # Alternatively, relax each pixel based on the *current* state of its neighbors.
    # We will update in-place as per standard iterative relaxation.
    
    # Pre-calculate offsets
    pixel_width = screen.pixel_width
    pixel_height = screen.pixel_height
    bounds = (0, pixel_width, 0, pixel_height)
    
    new_positions_list = []
    
    if indices is None:
        pixel_indices = [(i, j) for i in range(screen.nx) for j in range(screen.ny)]
    else:
        pixel_indices = indices

    for (i, j) in pixel_indices:
        target_pixel = screen.pixels[i][j]
        if len(target_pixel.positions) == 0:
            new_positions_list.append((i, j, target_pixel.positions))
            continue
                
        points = target_pixel.positions.copy()
        surrounding_points = []
        
        # Get padding points from surrounding pixels
        for (ni, nj) in screen.get_surrounding_indices((i, j)):
            surr_pixel = screen.pixels[ni][nj]
            if len(surr_pixel.positions) > 0:
                dx = (ni - i) * pixel_width
                dy = (nj - j) * pixel_height
                offset_positions = surr_pixel.positions + np.array([dx, dy])
                surrounding_points.append(offset_positions)
              
            # Perform Lloyd relaxation
            relaxed_all_points = Lloyd.lloyd_relaxation(points, bounds, alpha=alpha, iterations=iterations, padding = surrounding_points)
            
            # Extract the updated points for the target pixel
            updated_points = relaxed_all_points[:len(points)]
            new_positions_list.append((i, j, updated_points))
            
    # Apply all updates
    for i, j, new_pos in new_positions_list:
        screen.pixels[i][j].positions = new_pos
