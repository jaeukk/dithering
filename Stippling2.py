import numpy as np
import sys
import os
from scipy.interpolate import RegularGridInterpolator

# Import base class from Stippling.py
# Assuming Stippling.py is in the same directory.
try:
    from Stippling import BaseStipple
except ImportError:
    # Fallback for different execution environments
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from Stippling import BaseStipple

class MilisecondsStipple(BaseStipple):
    """
    @class MilisecondsStipple
    @brief Extends BaseStipple to implement real-time stippling algorithms.
    
    This class implements the "reverse color stippling" and "forward color stippling"
    algorithms as described in the paper "Milliseconds Color Stippling" (MM '21)
    by Lei Ma, Jian Shi, and Yanyun Chen.
    """

    def __init__(self, *args, **kwargs):
        """
        @brief Initialize MilisecondsStipple.
        @param args Positional arguments for BaseStipple.
        @param kwargs Keyword arguments for BaseStipple.
        """
        super().__init__(*args, **kwargs)
        self.type_to_color = {}

    def apply_reverse_stippling(self, points: np.ndarray, tone_levels: int = 3):
        """
        @brief Implementation of Reverse Multi-tone Color Stippling (Algorithm 2).
        
        This method is designed for placing light stipples on a black background.
        It is particularly effective for dark-colored images or regions to avoid 
        the overcrowding of dense dark stipples.
        
        @param points (N, 2) array of pre-generated sample points (Incremental Sample Sequence).
        @param tone_levels Number of tone levels 'L' per color channel. Default is 3.
        @return list of dictionaries containing stipple data (pos, radius, color, type).
        """
        if self.r_interp is None:
            self.create_rgb_interpolators()
            
        A0 = self.width * self.height
        n_candidates = len(points)
        delta = A0 / n_candidates  # Nominal area per sample (delta in the paper)
        
        g_levels = np.linspace(0, 1, tone_levels)
        
        # Clear existing pixels in the screen
        for i in range(self.screen.nx):
            for j in range(self.screen.ny):
                px = self.screen.pixels[i][j]
                px.positions = np.empty((0, 2), dtype=float)
                px.radii = np.empty(0, dtype=float)
                px.types = np.empty(0, dtype=int)

        color_to_type = {}
        stipples_data = []

        for i, p in enumerate(points):
            idx = i + 1  # 1-based order in the ISS
            x, y = p
            
            # Boundary check
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            
            # 1. Fetch fragment color (Algorithm 1, Line 9)
            # Interpolators expect (y, x) based on Stippling.py grid setup
            try:
                r_p = float(self.r_interp([y,x])[0])
                g_p = float(self.g_interp([y,x])[0])
                b_p = float(self.b_interp([y,x])[0])
            except (ValueError, IndexError):
                continue

            # 2. Adjust stipple size and fragment color (Algorithm 2)
            # Δδ ← max(rp, gp, bp)
            delta_delta = max(r_p, g_p, b_p)
            #delta_delta = min(r_p, g_p, b_p)  // This line gives more reasonable figure...
            
            # δi ← δ × (1 − Δδ)
            stipple_area = delta * (1.0 - delta_delta)
            stipple_area = max(0.0, stipple_area)
            radius = np.sqrt(stipple_area / np.pi)
            
            # cp ← cp / (1 − Δδ)
            denom = 1.0 - delta_delta
            if denom < 1e-6:
                # If area is zero, stipple color becomes irrelevant as area -> 0
                c_prime = np.array([r_p, g_p, b_p])
            else:
                c_prime = np.array([r_p, g_p, b_p]) / denom
                c_prime = np.clip(c_prime, 0, 1)

            # 3. Determine the stipple color (Algorithm 1, Lines 16-22)
            ai = A0 / idx
            final_rgb = []
            
            for val in c_prime:
                # Find l such that g_l <= val < g_{l+1}
                l_idx = np.searchsorted(g_levels, val) - 1
                l_idx = max(0, min(l_idx, tone_levels - 2))
                
                g_l = g_levels[l_idx]
                g_l_p1 = g_levels[l_idx + 1]
                
                # g* threshold (rejection threshold in the paper)
                # g* = delta * (g_{l+1} - g_l) / (channel_val - g_l)
                diff = val - g_l
                if diff < 1e-8:
                    tone = g_l
                else:
                    g_star = delta * (g_l_p1 - g_l) / diff
                    # ri <- g_{l+1} if ai >= r*, else g_l
                    if ai >= g_star:
                        tone = g_l_p1
                    else:
                        tone = g_l
                final_rgb.append(tone)
            
            color_tuple = tuple(final_rgb)
            if color_tuple not in color_to_type:
                color_to_type[color_tuple] = len(color_to_type)
            
            type_idx = color_to_type[color_tuple]
            
            # Store in stipples list
            stipple_info = {
                'pos': (x, y),
                'radius': radius,
                'color': color_tuple,
                'type': type_idx
            }
            stipples_data.append(stipple_info)
            
            # Add to Screen
            px_idx = self.screen.get_pixel_index((x, y))
            pixel = self.screen.pixels[px_idx[0]][px_idx[1]]
            
            if len(pixel.positions) == 0:
                pixel.positions = np.array([[x, y]])
                pixel.radii = np.array([radius])
                pixel.types = np.array([type_idx], dtype=int)
            else:
                pixel.positions = np.vstack([pixel.positions, [x, y]])
                pixel.radii = np.append(pixel.radii, radius)
                pixel.types = np.append(pixel.types, type_idx)

        # Store the color mapping for potential use in visualization
        self.type_to_color = {v: k for k, v in color_to_type.items()}
        
        return stipples_data

    def apply_forward_stippling(self, points: np.ndarray, tone_levels: int = 3):
        """
        @brief Implementation of Forward Multi-tone Color Stippling (Algorithm 1).
        
        This method places dark stipples on a white background.
        
        @param points (N, 2) array of pre-generated sample points (Incremental Sample Sequence).
        @param tone_levels Number of tone levels 'L' per color channel. Default is 3.
        @return list of dictionaries containing stipple data (pos, radius, color, type).
        """
        if self.r_interp is None:
            self.create_rgb_interpolators()
            
        A0 = self.width * self.height
        n_candidates = len(points)
        delta = A0 / n_candidates
        
        g_levels = np.linspace(0, 1, tone_levels)
        
        # Clear screen
        for i in range(self.screen.nx):
            for j in range(self.screen.ny):
                px = self.screen.pixels[i][j]
                px.positions = np.empty((0, 2), dtype=float)
                px.radii = np.empty(0, dtype=float)
                px.types = np.empty(0, dtype=int)

        color_to_type = {}
        stipples_data = []

        for i, p in enumerate(points):
            idx = i + 1
            x, y = p
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            
            try:
                r_p = float(self.r_interp([y, x])[0])
                g_p = float(self.g_interp([y, x])[0])
                b_p = float(self.b_interp([y, x])[0])
            except (ValueError, IndexError):
                continue

            # 2. Adjust stipple size and color (Algorithm 1)
            # Δδ ← min(rp, gp, bp)
            delta_delta = min(r_p, g_p, b_p)
            stipple_area = delta * (1.0 - delta_delta)
            stipple_area = max(0.0, stipple_area)
            radius = np.sqrt(stipple_area / np.pi)
            
            # Δc ← (Δδ, Δδ, Δδ); cp ← (cp − Δc) / (1 − Δδ)
            denom = 1.0 - delta_delta
            if denom < 1e-6:
                c_prime = np.array([0.0, 0.0, 0.0])
            else:
                c_prime = (np.array([r_p, g_p, b_p]) - delta_delta) / denom
                c_prime = np.clip(c_prime, 0, 1)

            # 3. Determine color
            ai = A0 / idx
            final_rgb = []
            for val in c_prime:
                l_idx = np.searchsorted(g_levels, val) - 1
                l_idx = max(0, min(l_idx, tone_levels - 2))
                g_l = g_levels[l_idx]
                g_l_p1 = g_levels[l_idx + 1]
                diff = val - g_l
                if diff < 1e-8:
                    tone = g_l
                else:
                    g_star = delta * (g_l_p1 - g_l) / diff
                    if ai >= g_star:
                        tone = g_l_p1
                    else:
                        tone = g_l
                final_rgb.append(tone)
            
            color_tuple = tuple(final_rgb)
            if color_tuple not in color_to_type:
                color_to_type[color_tuple] = len(color_to_type)
            type_idx = color_to_type[color_tuple]
            
            stipple_info = {'pos': (x, y), 'radius': radius, 'color': color_tuple, 'type': type_idx}
            stipples_data.append(stipple_info)
            
            px_idx = self.screen.get_pixel_index((x, y))
            pixel = self.screen.pixels[px_idx[0]][px_idx[1]]
            if len(pixel.positions) == 0:
                pixel.positions = np.array([[x, y]])
                pixel.radii = np.array([radius])
                pixel.types = np.array([type_idx], dtype=int)
            else:
                pixel.positions = np.vstack([pixel.positions, [x, y]])
                pixel.radii = np.append(pixel.radii, radius)
                pixel.types = np.append(pixel.types, type_idx)

        self.type_to_color = {v: k for k, v in color_to_type.items()}
        return stipples_data

    def get_styles(self):
        """
        @brief Helper to generate styles dictionary for Screen.visualize().
        @return list of dictionaries specifying styles for each particle type.
        """
        styles = []
        if not hasattr(self, 'type_to_color'):
            return styles
            
        for t in range(len(self.type_to_color)):
            color = self.type_to_color[t]
            styles.append({
                'facecolor': color,
                'edgecolor': 'none',
                'linewidth': 0
            })
        return styles
