import numpy as np

"""
CConversions.py
Collection of functions for color conversions between RGB and CIE XYZ / CIELAB.
Adheres to Adobe RGB (1998) Standard by default.
"""

# RGB to XYZ matrix for Adobe RGB (1998) with D65 white point
# Matrix values from Adobe RGB (1998) Color Image Encoding specification
RGB2XYZ = np.matrix([
    [0.57667, 0.18556, 0.18823],
    [0.29734, 0.62736, 0.07529],
    [0.02703, 0.07069, 0.99134]
])

# Inverse matrix: XYZ to RGB
XYZ2RGB = RGB2XYZ.I

# D65 Reference White (normalized to Y=1.0)
# Sum of columns/rows in the transformation matrix corresponds to the white point
WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])

def rgb_8bit_to_unit(rgb_8bit):
    """
    Converts 8-bit RGB values [0, 255] to unit values [0, 1].
    """
    return np.array(rgb_8bit) / 255.0

def unit_to_rgb_8bit(rgb_unit):
    """
    Converts unit RGB values [0, 1] to 8-bit values [0, 255].
    """
    return np.clip(np.round(np.array(rgb_unit) * 255.0), 0, 255).astype(np.uint8)

def remove_gamma(rgb_unit, gamma=2.19921875):
    """
    Removes gamma from unit RGB to obtain linearized RGB.
    Default gamma is 2.19921875 (Adobe RGB 1998 Standard).
    """
    return np.power(np.maximum(rgb_unit, 0), gamma)

def apply_gamma(rgb_linear, gamma=2.19921875):
    """
    Applies gamma to linearized RGB values.
    Default gamma is 2.19921875 (Adobe RGB 1998 Standard).
    """
    return np.power(np.maximum(rgb_linear, 0), 1.0 / gamma)

def linearized_rgb_to_xyz(rgb_linear):
    """
    Converts linearized RGB to CIE XYZ using the RGB2XYZ matrix.
    """
    # Handle both single vectors and arrays of vectors
    input_arr = np.array(rgb_linear)
    if input_arr.ndim == 1:
        return np.array(np.dot(RGB2XYZ, input_arr)).flatten()
    
    shape = input_arr.shape
    flat_rgb = input_arr.reshape(-1, 3).T
    xyz = np.dot(RGB2XYZ, flat_rgb)
    return np.array(xyz.T).reshape(shape)

def xyz_to_linearized_rgb(xyz):
    """
    Converts CIE XYZ to linearized RGB using the XYZ2RGB matrix.
    """
    input_arr = np.array(xyz)
    if input_arr.ndim == 1:
        return np.array(np.dot(XYZ2RGB, input_arr)).flatten()
    
    shape = input_arr.shape
    flat_xyz = input_arr.reshape(-1, 3).T
    rgb = np.dot(XYZ2RGB, flat_xyz)
    return np.array(rgb.T).reshape(shape)

def xyz_to_cielab(xyz, white_point=WHITE_D65):
    """
    Converts CIE XYZ to CIELAB space.
    """
    xyz = np.array(xyz)
    xyz_rel = xyz / white_point
    
    def f(t):
        delta = 6.0/29.0
        mask = t > (delta**3)
        res = np.zeros_like(t)
        res[mask] = np.power(t[mask], 1.0/3.0)
        res[~mask] = t[~mask] / (3.0 * delta**2) + (4.0/29.0)
        return res
    
    # Check if we have a single vector or an array
    if xyz.ndim == 1:
        fx, fy, fz = f(xyz_rel[0]), f(xyz_rel[1]), f(xyz_rel[2])
    else:
        fx, fy, fz = f(xyz_rel[..., 0]), f(xyz_rel[..., 1]), f(xyz_rel[..., 2])
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    if xyz.ndim == 1:
        return np.array([L, a, b])
    return np.stack([L, a, b], axis=-1)

def cielab_to_xyz(lab, white_point=WHITE_D65):
    """
    Converts CIELAB space back to CIE XYZ.
    """
    lab = np.array(lab)
    if lab.ndim == 1:
        L, a, b = lab[0], lab[1], lab[2]
    else:
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    
    def f_inv(t):
        delta = 6.0/29.0
        mask = t > delta
        res = np.zeros_like(t)
        res[mask] = np.power(t[mask], 3.0)
        res[~mask] = 3.0 * delta**2 * (t[~mask] - 4.0/29.0)
        return res
    
    x = f_inv(fx) * white_point[0]
    y = f_inv(fy) * white_point[1]
    z = f_inv(fz) * white_point[2]
    
    if lab.ndim == 1:
        return np.array([x, y, z])
    return np.stack([x, y, z], axis=-1)

def delta_e_cie76(lab1, lab2):
    """
    Calculates the CIELAB Delta E* color difference (CIE76).
    """
    return np.sqrt(np.sum(np.power(np.array(lab1) - np.array(lab2), 2), axis=-1))

def convert_from_preliminary_vectors(coeffs, basis_vectors):
    """
    Converts colors from a basis of three arbitrary preliminary vectors.
    
    Args:
        coeffs: (..., 3) array of coefficients for the basis.
        basis_vectors: (3, 3) array where each row is a basis vector 
                       in the target coordinate system (e.g., XYZ).
                       
    Returns:
        The combined color vector(s) in the target coordinate system.
    """
    # Result = sum(c_i * v_i) which is equivalent to coeffs @ basis_vectors
    return np.array(np.dot(coeffs, basis_vectors))
