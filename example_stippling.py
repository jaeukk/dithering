import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Adjust sys.path to ensure we can import the necessary modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add Pixels path
pixels_path = os.path.abspath(os.path.join(current_dir, '..', 'pixels'))
if pixels_path not in sys.path:
    sys.path.append(pixels_path)

from Stippling2 import MilisecondsStipple
import Lloyd

def main():
    """
    Example script to demonstrate forward and reverse multi-tone color stippling.
    """
    image_path = os.path.join(current_dir, 'data', 'color1_1.png')
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Loading image from {image_path}...")
    
    # 1. Initialize MilisecondsStipple
    # pixel_width/height control the 'grid' size for Screen object.
    # A larger pixel size means fewer Pixel objects in the Screen, but doesn't limit point resolution.
    sample = MilisecondsStipple(image_path, pixel_width=10.0, pixel_height=10.0)
    width, height = sample.width, sample.height
    print(f"Image dimensions: {width}x{height}")
    
    # 2. Generate an Incremental Sample Sequence (ISS)
    # The algorithms assume 'points' is a blue-noise distributed sequence.
    # Here we use Lloyd relaxation on random points to generate a uniform blue noise set.
    num_points = 8000
    print(f"Generating uniform sample sequence with {num_points} points...")
    np.random.seed(42)
    initial_points = np.random.uniform(0, [width, height], (num_points, 2))
    
    points = Lloyd.lloyd_relaxation(initial_points, (0, width, 0, height), alpha=1.0, iterations=5)
    
    # 3. Apply Forward Stippling (Algorithm 1)
    # Placing dark stipples on white background.
    print("Applying Forward Multi-tone Color Stippling...")
    sample.apply_forward_stippling(points, tone_levels=3)
    forward_screen = sample.screen.copy()
    forward_styles = sample.get_styles()
    
   
    # 5. Visualization
    print("Visualizing results...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    

    # Subplot 0: Forward Stippling
    axes[0].set_facecolor('white')
    axes[0].imshow(sample.image_data)
    axes[0].set_title("Original image")
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(0, height)
    axes[0].set_aspect('equal')

    # Subplot 0: Forward Stippling
    axes[1].set_facecolor('white')
    forward_screen.visualize([fig, axes[1]], forward_styles)
    axes[1].set_title("Forward Stippling (Dark on White) \n (Visualization via Patches)")
    axes[1].set_xlim(0, width)
    axes[1].set_ylim(0, height)
    axes[1].set_aspect('equal')
    #axes[0].invert_yaxis() # Match image coordinate system
    
    # Subplot 1: Reverse Stippling
    axes[2].set_facecolor('black')
    forward_screen.visualize([fig, axes[2]], forward_styles, use_scatter=True)
    axes[2].set_title("Forward Stippling (Dark on White) \n (Visualization via Dots)")
    axes[2].set_xlim(0, width)
    axes[2].set_ylim(0, height)
    axes[2].set_aspect('equal')
    #axes[1].invert_yaxis() # Match image coordinate system
    
    plt.tight_layout()
    output_fig = os.path.join(current_dir, 'stippling_demo.png')
    plt.savefig(output_fig, dpi=150)
    print(f"Demo result saved to {output_fig}")
    plt.show()

if __name__ == "__main__":
    main()
