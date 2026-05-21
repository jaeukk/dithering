import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import argparse

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

def parse_args():
    """
    Parse command-line arguments for the stippling script.
    """
    parser = argparse.ArgumentParser(description="Example script for sequential stippling using the SequentialStipple class.")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Path and Dimensions
    parser.add_argument('--image_path', type=str, default=os.path.join(current_dir, "data", "color1_2.png"),
                        help="Path to the input color image (default: data/color1_2.png).")
    parser.add_argument('--unit_width', type=float, default=25.0, help="Width of subdomin in pixels (default: 25.0).")
    parser.add_argument('--unit_height', type=float, default=25.0, help="Width of subdomin in pixels (default: 25.0).")
    parser.add_argument('--output_path', type=str, default=os.path.join(current_dir, "example_stippling3_output.png"),
                        help="Path to save the output visualization (default: example_stippling3_output.png).")
    
    # 2. Sampling and Relaxation Parameters
    parser.add_argument('--radius', type=float, default=0.03, help="Normalized radius relative to pixel width (default: 0.03).")
    parser.add_argument('--seed', type=int, default=44, help="Seed for random number generation (default: 44).")
    parser.add_argument('--standard', type=str, default='luminance', 
                        choices=['luminance', 'inverse_luminance', 'grayscale', 'uniform'],
                        help="Density mapping standard for initial sampling (default: luminance).")
    parser.add_argument('--relax_local_iter', type=int, default=20, help="Iterations for local relaxation (default: 20).")
    parser.add_argument('--relax_global_iter', type=int, default=50, help="Iterations for global relaxation (default: 50).")
    
    # 3. Color Assignment Parameters
    parser.add_argument('--method', type=str, default='Floyd-Steinberg',
                        choices=['arithematic', 'area_weighted', 'Floyd-Steinberg'],
                        help="Method for assigning final colors (default: Floyd-Steinberg).")
    parser.add_argument('--error', type=float, default=1.0, help="Error threshold for color assignment (default: 1.0).")
    parser.add_argument('--rnd_points', type=int, default=1600, help="Number of random points for cell color averaging (default: 200).")

    return parser.parse_args()

def main():
    """
    Example script for sequential stippling using the SequentialStipple class.
    Refactored from demo2.py.
    """
    args = parse_args()

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
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    # unit_width and unit_height define the size of each pixel in physical units (e.g., mm)
    stippler = SequentialStipple(args.image_path, pixel_width=args.unit_width, pixel_height=args.unit_height)
    stippler.set_dome_library(dome_library)
    
    # 3. Execute the full stippling pipeline
    print("Step 1: Setting radius...")
    # The radius is a normalized value relative to the pixel width
    stippler.set_radius(args.radius)
    
    print("Step 2: Creating initial sampling points...")
    # Uses rejection sampling based on a specified density mapping.
    # Options for 'standard':
    #   - 'luminance' (default): Uses the Y channel (relative luminance). Points cluster in darker areas.
    #   - 'inverse_luminance': Uses 1.0 - Y channel. Points cluster in lighter areas.
    #   - 'grayscale': Uses standard Luma (0.299R + 0.587G + 0.114B) directly from RGB image data.
    #   - 'uniform': Applies a flat, constant density across the entire image space.
    stippler.create_initial_sampling_points(seed=args.seed, standard=args.standard)
    
    print("Step 3: Performing relaxation of sampling points (individual pixels)...")
    # Aligns points locally within each pixel to improve distribution
    stippler.relax_individual_pixels(iterations=args.relax_local_iter)
    # Performs Lloyd relaxation across all points to achieve a global uniform distribution
    stippler.relax_entire_screen(iterations=args.relax_global_iter)
    
    print("Step 4: Assigning final colors...")
    # Calculates the average color within each Voronoi cell
    stippler.get_average_colors(rnd_points=args.rnd_points)
    
    # Assigns the best-matching dome type from the library to each sampling point.
    # Options for 'method':
    #   - 'arithematic' (default): Assigns the closest color locally, and balances remaining error by uniformly averaging with neighbors.
    #   - 'area_weighted': Similar to arithematic but inverse-weights the color mixing iteratively based on Voronoi cell areas.
    #   - 'Floyd-Steinberg': Adapts classic Floyd-Steinberg error diffusion onto the unstructured Voronoi grid. 
    #                        Scans cells top-to-bottom and diffuses RGB quantization errors to neighboring unprocessed cells.
    stippler.assign_colors(error=args.error, method=args.method)
    
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
    fig.savefig(args.output_path, bbox_inches='tight', pad_inches=0, dpi = 1000)
    print(f"Result successfully saved to {args.output_path}")

if __name__ == "__main__":
    main()
