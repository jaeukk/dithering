import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the current directory to sys.path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Stippling2 import MilisecondsStipple

def main():
    """
    @brief Script to visualize and verify RGB interpolators in MilisecondsStipple.
    """
    image_path = os.path.join(current_dir, 'data', 'color1_1.png')
    output_path = os.path.join(current_dir, 'interpolator_check.png')
    
    print(f"Loading image from: {image_path}")
    
    # 1. & 2. Initialize MilisecondsStipple with the image
    # This automatically calls input_image and create_rgb_interpolators
    sample = MilisecondsStipple(image_path=image_path)
    
    # 3. Verify or create the RGB interpolators
    if sample.r_interp is None:
        print("Creating RGB interpolators...")
        sample.create_rgb_interpolators()
    else:
        print("RGB interpolators already initialized.")
        
    # 5. Reconstruct the channels from interpolators
    # Generate a grid of points (x, y) over the full width and height of the sample.
    w, h = sample.width, sample.height
    ny, nx, _ = sample.image_data.shape
    
    print(f"Image dimensions: {nx}x{ny} (Width x Height: {w}x{h})")
    
    # Create coordinate arrays matching the interpolator's grid
    x_coords = np.linspace(0, w, nx)
    y_coords = np.linspace(0, h, ny)
    
    # Create meshgrid for evaluation
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # RegularGridInterpolator expects (y, x) points
    eval_points = np.stack([Y.ravel(), X.ravel()], axis=-1)
    
    print("Evaluating interpolators on grid...")
    r_reconstructed = sample.r_interp(eval_points).reshape(ny, nx)
    g_reconstructed = sample.g_interp(eval_points).reshape(ny, nx)
    b_reconstructed = sample.b_interp(eval_points).reshape(ny, nx)
    
    # 4. Create a 2D visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original Image (Top-Down orientation)
    axes[0, 0].imshow(sample.image_data)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Reconstructed Channels
    # Since interpolators use flipped_data (y=0 at bottom), 
    # and our Y meshgrid has y[0]=0, r_reconstructed[0,:] corresponds to the bottom row.
    # Using origin='lower' in imshow will place index 0 at the bottom, 
    # matching the original image's bottom row at the bottom of the plot.
    
    axes[0, 1].imshow(r_reconstructed, cmap='Reds', origin='lower')
    axes[0, 1].set_title("Reconstructed Red Channel")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(g_reconstructed, cmap='Greens', origin='lower')
    axes[1, 0].set_title("Reconstructed Green Channel")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(b_reconstructed, cmap='Blues', origin='lower')
    axes[1, 1].set_title("Reconstructed Blue Channel")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to: {output_path}")

if __name__ == "__main__":
    main()
