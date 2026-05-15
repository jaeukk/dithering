import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

# Set plotting styles for professional output
matplotlib.rcParams.update({'font.size': 15, "text.usetex": True, "font.family": "Times New Roman"})
matplotlib.rcParams.update({"xtick.direction":"in", "xtick.top":True, "ytick.direction":"in", "ytick.right":True})
plt.rcParams['text.latex.preamble'] = r'\usepackage{xcolor}'

# Add current directory to sys.path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Stippling3 import SequentialStipple
import CConversions as cc

def main():
    """
    Example script for sequential stippling using the SequentialStipple class.
    Refactored from demo2.py.
    """
    # 1. Define the dome library (RGB and radius values from demo2.py)
    # These represent different types of 'stipples' (domes) with their physical properties.
    dome_library = [
        {"rgb": [99, 200, 138], "radius": 13.0},
        {"rgb": [201, 117, 151], "radius": 16.0},
        {"rgb": [56, 96, 202], "radius": 11.0}
    ]
    
    # Precompute XYZ and LAB color spaces for the dome library for faster processing
    for dt in dome_library:
        unit_rgb = cc.rgb_8bit_to_unit(dt["rgb"])
        xyz = cc.linearized_rgb_to_xyz(unit_rgb)
        lab = cc.xyz_to_cielab(xyz)
        dt["xyz"] = xyz
        dt["lab"] = lab

    # 2. Initialize SequentialStipple with a color image
    image_path = os.path.join(current_dir, "data", "color1_2.png")
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # unit_width and unit_height define the size of each pixel in physical units (e.g., mm)
    unit_width, unit_height = 25.0, 25.0
    stippler = SequentialStipple(image_path, pixel_width=unit_width, pixel_height=unit_height)
    stippler.set_dome_library(dome_library)
    
    # 3. Execute the full stippling pipeline
    print("Step 1: Setting radius...")
    # The radius is a normalized value relative to the pixel width
    stippler.set_radius(0.1)
    
    print("Step 2: Creating initial sampling points...")
    # Uses rejection sampling based on the luminance of the image
    stippler.create_initial_sampling_points(seed=44)
    
    print("Step 3: Performing local relaxation (individual pixels)...")
    # Aligns points locally within each pixel to improve distribution
    stippler.relax_individual_pixels(iterations=20)
    
    print("Step 4: Performing global relaxation (entire screen)...")
    # Performs Lloyd relaxation across all points to achieve a global uniform distribution
    stippler.relax_entire_screen(iterations=50)
    
    print("Step 5: Computing average colors for cells...")
    # Calculates the average color within each Voronoi cell
    stippler.get_average_colors(rnd_points=200)
    
    print("Step 6: Assigning final colors...")
    # Assigns the best-matching dome type from the library to each sampling point
    stippler.assign_colors(error=0.1)
    
    # 4. Visualize and save the result
    print("Generating visualization...")
    fig = plt.figure(figsize=(8, 8 * stippler.height / stippler.width))
    ax = fig.add_axes([0,0,1,1])
    ax.set_facecolor("k")
    
    # Define styles for each dome type based on their RGB values
    styles = [{'facecolor': np.array(dt['rgb'])/255.0, 'edgecolor': 'none', 'linewidth': 0} for dt in dome_library]
    
    # Use the screen's visualization method with the calculated length2pixel factor
    stippler.screen.visualize([fig, ax], styles, use_collection=True, radius_to_pixel=1.0)
    
    fig.tight_layout()
    output_path = os.path.join(current_dir, "example_stippling3_output.png")
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi = 1000)
    print(f"Result successfully saved to {output_path}")

if __name__ == "__main__":
    main()
