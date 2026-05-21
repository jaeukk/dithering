# Dithering and Stippling

This directory contains a collection of Python tools for generating stippled images using Lloyd's relaxation and multi-tone color stippling algorithms.

## Core Modules

### 1. [`Lloyd.py`](Lloyd.py)
Implements algorithms for point distribution and relaxation:
- **Ordinary Lloyd Relaxation**: Standard Centroidal Voronoi Tessellation (CVT) update for uniform point distribution.
- **Modified Lloyd Relaxation**: CVT update weighted by a density function $\rho(x, y)$, allowing points to cluster in regions of higher density (e.g., darker areas of an image).

### 2. [`Stippling.py`](Stippling.py)
Defines the `BaseStipple` class for image processing:
- **Image Input**: Loads and normalizes images to $[0, 1]$ range.
- **RGB Interpolators**: Uses `scipy.interpolate.RegularGridInterpolator` for efficient retrieval of color data at arbitrary coordinates.
- **HSV & Density Mapping**: Methods to create interpolators based on hue, saturation, value, and calculated color density.

### 3. [`Stippling2.py`](Stippling2.py)
Extends the base functionality with the `MilisecondsStipple` class, implementing algorithms from the paper *"Milliseconds Color Stippling"* (MM '21) by Lei Ma, Jian Shi, and Yanyun Chen:
- **Forward Multi-tone Color Stippling**: Optimized for placing dark stipples on a white background.
- **Reverse Multi-tone Color Stippling**: Optimized for placing light stipples on a black background, preventing overcrowding in dark regions.

### 4. [`Stippling3.py`](Stippling3.py)
Introduces the `SequentialStipple` class, refactored for sequential and multi-toned dithering tasks utilizing customizable, pre-defined dome libraries (palettes).
- **Dynamic Density Mapping**: Features options for generating density maps from luminance, inverse luminance, grayscale, or uniformly.
- **Color Assignment & Error Diffusion**: Includes multiple quantization methods (`arithematic`, `area_weighted`, and `Floyd-Steinberg`) to intelligently allocate palette colors onto unstructured scattered Voronoi cells while diffusing color discrepancies contextually.

## Usage and Examples

- **[`demo.ipynb`](demo.ipynb)**: Interactive demonstration of Lloyd relaxation with both synthetic (Gaussian) and image-based density functions.
- **[`example_stippling2.py`](example_stippling2.py)**: A complete script demonstrating the application of forward and reverse stippling on a color image, including visualization.
- **[`example_stippling3.py`](example_stippling3.py)**: A comprehensive script executing the sequential multi-toned stippling workflow. Demonstrates density-based rejection sampling alongside Voronoi-based color assignment via advanced dithering methods like Floyd-Steinberg error diffusion.

## Requirements

- `numpy`
- `scipy`
- `Pillow`
- `matplotlib`
