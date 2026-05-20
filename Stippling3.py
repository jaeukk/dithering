import numpy as np
import scipy.interpolate
from scipy.spatial import Voronoi
from matplotlib.path import Path
import os
import sys
import pickle
import colorsys

# Adjust sys.path to import Pixels and Lloyd, CConversions
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Stippling import BaseStipple
from Lloyd import modified_lloyd_relaxation, generate_padding_pts
import CConversions as cc

class SequentialStipple(BaseStipple):
    """
    A class to handle sequential stippling process, refactored from demo2.py.
    """

    def __init__(self, image_path=None, width=None, height=None, nx=None, ny=None, pixel_width=1.0, pixel_height=1.0):
        super().__init__(image_path, width, height, nx, ny, pixel_width, pixel_height)
        self.max_area_fraction = 0.50 # Default from demo2.py
        self.length2pixel = 1.0
        self.radius = 0.03 # Default from demo2.py
        self.max_num_density = 0.0
        self.density_interp = None
        self.voronoi_tessellation = None
        self.padding_pts = None
        self.cell_data = None
        self.pt_map = None
        self.all_pts_global = None
        self.scale = None

        if self.image_data is not None:
            self._create_density_interpolator()

    def _get_xyz_data(self):
        """
        Convert image data to XYZ format and return as (width, height, 3) array.
        """
        ni, nj = self.image_data.shape[0], self.image_data.shape[1]
        return np.array([[cc.linearized_rgb_to_xyz(cc.remove_gamma(self.image_data[i, j])) 
                          for i in range(ni)] for j in range(nj)])

    def _create_interpolator(self, data):
        """
        Create a RegularGridInterpolator for the given data (nj, ni, ...).
        """
        nj, ni = data.shape[0], data.shape[1]
        x_coords = np.linspace(0, self.width, nj)
        y_coords = np.linspace(0, self.height, ni)
        return scipy.interpolate.RegularGridInterpolator(
            (x_coords, y_coords), np.flip(data, axis=1),
            bounds_error=False, fill_value=0
        )

    def _create_density_interpolator(self, mode="luminance"):
        """
        Create density interpolator of an input image based on XYZ luminance or other modes.
        """
        if self.image_data is None:
            return

        xyz_data = self._get_xyz_data()
        
        if mode == "luminance":
            # Luminance is the Y channel (index 1)
            density_funct = xyz_data[:, :, 1]
        elif mode == "uniform":
            density_funct = np.ones((xyz_data.shape[0], xyz_data.shape[1]))
        elif mode == "inverse_luminance":
            density_funct = 1.0 - xyz_data[:, :, 1]
        elif mode == "grayscale":
            # Standard luma (0.299R + 0.587G + 0.114B) from the original image_data
            gray = 0.299 * self.image_data[:, :, 0] + 0.587 * self.image_data[:, :, 1] + 0.114 * self.image_data[:, :, 2]
            density_funct = gray.T  # Transpose (ni, nj) to (nj, ni) to align with xyz_data mapping
        else:
            raise ValueError(f"Unknown density mode: {mode}")
            
        self.density_interp = self._create_interpolator(density_funct)

    def set_radius(self, radius):
        """
        Set normalized radius and update related parameters. max_radius / unit_width
        """
        self.radius = radius
        if self.dome_library:
            max_radius = max(dt["radius"] for dt in self.dome_library)
        else:
            max_radius = 1.0 # Fallback
            
        self.scale = self.radius * self.screen.pixel_width / max_radius
        self.length2pixel = max_radius / (self.screen.pixel_width * self.radius)
        self.max_num_density = self.max_area_fraction / np.pi / self.radius**2 / (self.screen.pixel_width * self.screen.pixel_height)

    def _rho_func(self, x, y):
        pts = np.column_stack((x, y))
        return self.density_interp(pts)

    def create_initial_sampling_points(self, seed=44, standard='luminance'):
        """
        Create initial sampling points using rejection sampling (demo2.py lines 126-159).
        """
        if standard is not None:
            self._create_density_interpolator(mode=standard)
            
        np.random.seed(seed)
        nx, ny = self.screen.nx, self.screen.ny
        uw, uh = self.screen.pixel_width, self.screen.pixel_height
        
        for i in range(nx):
            for j in range(ny):
                pixel = self.screen.pixels[i][j]
                x0, x1 = i * uw, (i + 1) * uw
                y0, y1 = j * uh, (j + 1) * uh
                
                # Estimate integral of density over pixel area using a sample grid
                test_x = np.linspace(x0, x1, 5)
                test_y = np.linspace(y0, y1, 5)
                xv, yv = np.meshgrid(test_x, test_y)
                avg_rho = np.mean(self._rho_func(xv.ravel(), yv.ravel()))
                integral = avg_rho * (uw * uh)
                
                predicted_n = int(round(integral * self.max_num_density))
                dots = []
                attempts = 0
                max_attempts = predicted_n * 100
                while len(dots) < predicted_n and attempts < max_attempts:
                    rx = np.random.uniform(x0, x1)
                    ry = np.random.uniform(y0, y1)
                    if np.random.random() < self._rho_func(rx, ry)[0]:
                        dots.append([rx - x0, ry - y0])
                    attempts += 1
                    
                if dots:
                    pixel.positions = np.array(dots)
                    pixel.radii = np.ones(len(dots)) * (self.radius * uw)
                    pixel.types = np.zeros(len(dots), dtype=int)

    def relax_individual_pixels(self, iterations=20):
        """
        Relax the sampling points of individual pixels (demo2.py lines 177-207).
        """
        nx, ny = self.screen.nx, self.screen.ny
        uw, uh = self.screen.pixel_width, self.screen.pixel_height
        
        for i in range(nx):
            for j in range(ny):
                pixel = self.screen.pixels[i][j]
                if len(pixel.positions) == 0:
                    continue
                    
                x0, y0 = i * uw, j * uh
                bounds = (x0, x0 + uw, y0, y0 + uh)
                
                # Gather neighboring points for padding
                padding_pts = []
                for ni_idx, nj_idx in self.screen.get_surrounding_indices((i, j)):
                    neighbor = self.screen.pixels[ni_idx][nj_idx]
                    if len(neighbor.positions) > 0:
                        off_x, off_y = ni_idx * uw, nj_idx * uh
                        padding_pts.append(neighbor.positions + np.array([off_x, off_y]))      
                padding = np.vstack(padding_pts) if padding_pts else None
                
                # Global position for relaxation
                pts_global = pixel.positions + np.array([x0, y0])
                
                # Apply relaxation
                relaxed_global = modified_lloyd_relaxation(
                    pts_global, bounds, rho=self._rho_func, 
                    alpha=1., iterations=iterations, samp_pts=1600, padding_pts=padding
                )
                
                # Replace
                pixel.positions = relaxed_global - np.array([x0, y0])

    def relax_entire_screen(self, iterations=10):
        """
        Relax the entire screen (demo2.py lines 215-252).
        """
        nx, ny = self.screen.nx, self.screen.ny
        uw, uh = self.screen.pixel_width, self.screen.pixel_height
        width, height = self.width, self.height
        
        # Count total points and gather all points
        all_pts = []
        for i in range(nx):
            for j in range(ny):
                if len(self.screen.pixels[i][j].positions) > 0:
                    all_pts.append(self.screen.pixels[i][j].positions + np.array([i * uw, j * uh]))

        if not all_pts:
            return

        all_pts = np.vstack(all_pts)
        N_tot = len(all_pts)

        padding_pts = generate_padding_pts((0, width, 0, height), int(round(4.2 * np.sqrt(N_tot))), padding_factor=0.001)
        self.padding_pts = padding_pts

        # Apply relaxation exactly once with iterations
        relaxed_all, self.voronoi_tessellation = modified_lloyd_relaxation(
            all_pts, (0, width, 0, height), rho=self._rho_func, 
            alpha=1., iterations=iterations, samp_pts=1600, padding_pts=padding_pts,
            return_voronoi=True
        )
        self.all_pts_global = relaxed_all
        self.pt_map = [None] * len(relaxed_all)
        
        # Clear and redistribute
        for i in range(nx):
            for j in range(ny):
                self.screen.pixels[i][j].positions = np.empty((0, 2))
                self.screen.pixels[i][j].types = np.empty(0, dtype=int)
                self.screen.pixels[i][j].radii = np.empty(0)

        for k, p in enumerate(relaxed_all):
            idx = self.screen.get_pixel_index(p)
            px_obj = self.screen.pixels[idx[0]][idx[1]]
            local_p = p - np.array([idx[0] * uw, idx[1] * uh])
            
            local_idx = len(px_obj.positions)
            self.pt_map[k] = (idx[0], idx[1], local_idx)
            
            if local_idx == 0:
                px_obj.positions = np.array([local_p])
                px_obj.radii = np.array([self.radius * uw])
                px_obj.types = np.array([0], dtype=int)
            else:
                px_obj.positions = np.vstack([px_obj.positions, local_p])
                px_obj.radii = np.append(px_obj.radii, self.radius * uw)
                px_obj.types = np.append(px_obj.types, 0)

    def get_average_colors(self, rnd_points=200):
        """
        Analyze geometric and colorimetric properties of Voronoi cells.
        """
        nx, ny = self.screen.nx, self.screen.ny
        uw, uh = self.screen.pixel_width, self.screen.pixel_height
        width, height = self.width, self.height

        # 1. Gather or Use existing global points and pt_map
        if self.all_pts_global is not None and self.pt_map is not None:
            all_pts_global = self.all_pts_global
        else:
            all_pts_global = []
            self.pt_map = [] # (pixel_i, pixel_j, local_index)
            for i in range(nx):
                for j in range(ny):
                    pixel = self.screen.pixels[i][j]
                    for idx, p in enumerate(pixel.positions):
                        all_pts_global.append(p + np.array([i * uw, j * uh]))
                        self.pt_map.append((i, j, idx))
            
            if not all_pts_global:
                self.cell_data = []
                return
            all_pts_global = np.array(all_pts_global)
            self.all_pts_global = all_pts_global
            self.voronoi_tessellation = None # Force re-computation to ensure consistency

        # 2. Ensure voronoi_tessellation is available
        if self.voronoi_tessellation is None:
            N_tot = len(all_pts_global)
            padding_pts = generate_padding_pts((0, width, 0, height), int(round(4.2 * np.sqrt(N_tot))), padding_factor=0.001)
            all_vor_pts = np.vstack([all_pts_global, padding_pts])
            self.voronoi_tessellation = Voronoi(all_vor_pts)
        
        vor = self.voronoi_tessellation

        # 3. Create XYZ interpolator
        xyz_data = self._get_xyz_data()
        xyz_interp = self._create_interpolator(xyz_data)

        def get_xyz(x, y):
            pts = np.column_stack((x, y))
            return xyz_interp(pts)

        # 4. Calculate cell areas, average XYZ/LAB, and neighbors
        self.cell_data = []
        for idx in range(len(all_pts_global)):
            region_idx = vor.point_region[idx]
            vertices = vor.vertices[vor.regions[region_idx]]
            
            # Area
            x = vertices[:, 0]; y = vertices[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            # Color
            v_min, v_max = np.min(vertices, axis=0), np.max(vertices, axis=0)
            sample_candidates = np.random.uniform(v_min, v_max, (rnd_points * 2, 2))
            path = Path(vertices)
            inside = path.contains_points(sample_candidates)
            sampled_pts = sample_candidates[inside][:rnd_points]
            
            if len(sampled_pts) > 0:
                cell_xyz = np.mean(get_xyz(sampled_pts[:, 0], sampled_pts[:, 1]), axis=0)
            else:
                cell_xyz = get_xyz([all_pts_global[idx][0]], [all_pts_global[idx][1]])[0]
            
            self.cell_data.append({
                'area': area,
                'xyz': cell_xyz,
                'lab': cc.xyz_to_cielab(cell_xyz),
                'assigned_type': -1,
                'neighbors': []
            })

        # Neighbors
        for p1, p2 in vor.ridge_points:
            if p1 < len(all_pts_global) and p2 < len(all_pts_global):
                self.cell_data[p1]['neighbors'].append(p2)
                self.cell_data[p2]['neighbors'].append(p1)

    # methods of color mixing
    def _arithematic(self, error):
        # colors are mixed arithematically in the nearest neighbors.
        max_radius = max(dt["radius"] for dt in self.dome_library)
        for idx, cell in enumerate(self.cell_data):
            target_lab = cell['lab']
            
            best_match_idx = -1
            min_dist = float('inf')
            
            #corr_factor = np.pi * (max_radius / self.length2pixel)**2 / cell['area'] / self.max_area_fraction
            corr_factor = 1.
            for t_idx, d_type in enumerate(self.dome_library):
                dist = cc.delta_e_cie76(target_lab, d_type['lab'] * corr_factor)
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = t_idx
            
            # Store assigned type if close enough, else store encoded negative index for refinement
            cell['assigned_type'] = best_match_idx if min_dist < error else -1 * (best_match_idx + 2)

        # 2. Refine unassigned cells by considering neighbors
        for idx, cell in enumerate(self.cell_data):
            # Recalculate corr_factor for the current cell as it depends on cell['area']
            #corr_factor = np.pi * (max_radius / self.length2pixel)**2 / cell['area'] / self.max_area_fraction
            corr_factor = 1
            
            if cell['assigned_type'] < -1:
                # Recover the initial best match from encoded negative index
                initial_best_t = -cell['assigned_type'] - 2
                
                # Identify assigned neighbors to balance color
                neigh_indices = [n for n in cell['neighbors'] if self.cell_data[n]['assigned_type'] >= 0]
                
                if not neigh_indices:
                    assigned_type = initial_best_t
                else:
                    total_area = sum(self.cell_data[n]['area'] for n in neigh_indices)
                    # Correctly decode neighbor's encoded type before indexing dome_library
                    mean_neigh_xyz = sum(self.dome_library[self.cell_data[n]['assigned_type']]['xyz'] for n in neigh_indices) * corr_factor / total_area
                    
                    best_t = -1
                    min_err = float('inf')
                    for t_idx, d_type in enumerate(self.dome_library):
                        new_mean_xyz = (mean_neigh_xyz * total_area + d_type['xyz'] * corr_factor) / (total_area + cell['area'])
                        err = np.linalg.norm(new_mean_xyz - cell['xyz'])
                        if err < min_err:
                            min_err = err
                            best_t = t_idx
                    assigned_type = best_t
                
                cell['assigned_type'] = assigned_type
            
            # Ensure assigned_type is correctly set for already assigned cells
            assigned_type = cell['assigned_type']
            
            # Update the pixel object with final results
            pixel_i, pixel_j, local_idx = self.pt_map[idx]
            pixel = self.screen.pixels[pixel_i][pixel_j]
            pixel.types[local_idx] = assigned_type
            pixel.radii[local_idx] = self.dome_library[assigned_type]['radius'] * self.scale

    def _area_weighted(self, error):
        # When mixing colors, dome colors are mixed inversely weighted by cell areas.

        # 1. Assign the perfect matches
        max_radius = max(dt["radius"] for dt in self.dome_library)
        for idx, cell in enumerate(self.cell_data):
            target_lab = cell['lab']
            
            best_match_idx = -1
            min_dist = float('inf')
            
            corr_factor = np.pi * (max_radius / self.length2pixel)**2 / cell['area'] / self.max_area_fraction
            for t_idx, d_type in enumerate(self.dome_library):
                dist = cc.delta_e_cie76(target_lab, d_type['lab'] * corr_factor)
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = t_idx
            
            # Store assigned type if close enough, else store encoded negative index for refinement
            cell['assigned_type'] = best_match_idx if min_dist < error else -1 * (best_match_idx + 2)

        # 2. Refine unassigned cells by considering neighbors
        for idx, cell in enumerate(self.cell_data):
            # Recalculate corr_factor for the current cell as it depends on cell['area']
            corr_factor = np.pi * (max_radius / self.length2pixel)**2 / cell['area'] / self.max_area_fraction
            
            if cell['assigned_type'] < -1:
                # Recover the initial best match from encoded negative index
                initial_best_t = -cell['assigned_type'] - 2
                
                # Identify assigned neighbors to balance color
                neigh_indices = [n for n in cell['neighbors'] if self.cell_data[n]['assigned_type'] >= 0]
                
                if not neigh_indices:
                    assigned_type = initial_best_t
                else:
                    total_area = sum(self.cell_data[n]['area'] for n in neigh_indices)
                    # Correctly decode neighbor's encoded type before indexing dome_library
                    mean_neigh_xyz = sum(self.dome_library[self.cell_data[n]['assigned_type']]['xyz'] for n in neigh_indices) * corr_factor / total_area
                    
                    best_t = -1
                    min_err = float('inf')
                    for t_idx, d_type in enumerate(self.dome_library):
                        new_mean_xyz = (mean_neigh_xyz * total_area + d_type['xyz'] * corr_factor) / (total_area + cell['area'])
                        err = np.linalg.norm(new_mean_xyz - cell['xyz'])
                        if err < min_err:
                            min_err = err
                            best_t = t_idx
                    assigned_type = best_t
                
                cell['assigned_type'] = assigned_type
            
            # Ensure assigned_type is correctly set for already assigned cells
            assigned_type = cell['assigned_type']
            
            # Update the pixel object with final results
            pixel_i, pixel_j, local_idx = self.pt_map[idx]
            pixel = self.screen.pixels[pixel_i][pixel_j]
            pixel.types[local_idx] = assigned_type
            pixel.radii[local_idx] = self.dome_library[assigned_type]['radius'] * self.scale

    def _Floyd_Steinberg(self, error):
        # Floyd-Steinberg error diffusion on Voronoi cells.
        max_radius = max(dt["radius"] for dt in self.dome_library)
        N = len(self.all_pts_global)
        
        if N == 0:
            return

        # Prepare working colors in XYZ space
        work_xyz = np.zeros((N, 3))
        for i, cell in enumerate(self.cell_data):
            work_xyz[i] = cell['xyz'].copy()
            
        # Build scan order (Serpentine based on Y coordinate)
        pts = self.all_pts_global
        area_avg = np.mean([c['area'] for c in self.cell_data])
        row_step = np.sqrt(area_avg) if area_avg > 0 else 1.0
        
        row_indices = np.floor(pts[:, 1] / row_step).astype(int)
        scan_order = []
        for r in range(np.min(row_indices), np.max(row_indices) + 1):
            in_row = np.where(row_indices == r)[0]
            if len(in_row) == 0:
                continue
            sorted_in_row = in_row[np.argsort(pts[in_row, 0])]
            if r % 2 == 1:
                sorted_in_row = sorted_in_row[::-1]
            scan_order.extend(sorted_in_row)

        processed = np.zeros(N, dtype=bool)

        for idx in scan_order:
            cell = self.cell_data[idx]
            
            # Clip current XYZ to valid range [0, 1] before processing
            current_xyz = np.clip(work_xyz[idx], 0.0, 1.0)
            current_lab = cc.xyz_to_cielab(current_xyz)
            
            best_match_idx = -1
            min_dist = float('inf')
            
            # Use corr_factor = 1.0 (similar to _arithematic)
            corr_factor = 1.0
            
            for t_idx, d_type in enumerate(self.dome_library):
                dist = cc.delta_e_cie76(current_lab, d_type['lab'] * corr_factor)
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = t_idx
            
            cell['assigned_type'] = best_match_idx
            processed[idx] = True
            
            chosen_xyz = self.dome_library[best_match_idx]['xyz'] * corr_factor
            err = current_xyz - chosen_xyz
            
            # Diffuse error to unprocessed neighbors
            unprocessed_neighbors = [nb for nb in cell['neighbors'] if not processed[nb]]
            if unprocessed_neighbors:
                w = 1.0 / len(unprocessed_neighbors)
                for nb in unprocessed_neighbors:
                    work_xyz[nb] = np.clip(work_xyz[nb] + w * err, 0.0, 1.0)

            # Update the pixel object with final results
            pixel_i, pixel_j, local_idx = self.pt_map[idx]
            pixel = self.screen.pixels[pixel_i][pixel_j]
            pixel.types[local_idx] = best_match_idx
            pixel.radii[local_idx] = self.dome_library[best_match_idx]['radius'] * self.scale


    def assign_colors(self, error=0.1, method = 'arithematic'):
        """
        Assign dome types to sampling points based on cell_data.
        """
        if not self.dome_library:
            raise ValueError("dome_library must be set before calling assign_colors.")
        if self.cell_data is None or self.pt_map is None:
            raise ValueError("get_average_colors must be called before assign_colors.")
        
        # Type Assignment (Sequential)
        # 1. Assign the perfect matches
        if method == "arithematic":
            self._arithematic(error)
        elif method == "area_weighted":
            self._area_weighted(error)
        elif method == "Floyd-Steinberg":
            self._Floyd_Steinberg(error)


    def save(self, path):
        """
        Save selected attributes and the screen object to a file using pickle.
        """
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Attributes to save (lines 25-35 and screen)
        attrs_to_save = [
            'max_area_fraction', 'length2pixel', 'radius', 'max_num_density',
            'density_interp', 'voronoi_tessellation', 'padding_pts',
            'cell_data', 'pt_map', 'all_pts_global', 'scale', 'screen'
        ]
        
        save_dict = {attr: getattr(self, attr) for attr in attrs_to_save if hasattr(self, attr)}

        try:
            with open(path, 'wb') as f:
                pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise IOError(f"Failed to save SequentialStipple state to {path}: {e}")

    def load(self, path):
        """
        Load selected attributes and the screen object from a file using pickle.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            with open(path, 'rb') as f:
                save_dict = pickle.load(f)
            
            if not isinstance(save_dict, dict):
                raise TypeError(f"Loaded object is of type {type(save_dict)}, expected dict")
            
            for attr, value in save_dict.items():
                setattr(self, attr, value)
            
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to load SequentialStipple state from {path}: {e}")
