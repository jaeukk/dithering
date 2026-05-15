import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.qhull import QhullError

def lloyd_relaxation(points, bounds, alpha=0.2, iterations=1, n_padding=10):
    """
    Performs ordinary Lloyd relaxation on a set of points.
    
    This is a special case of modified Lloyd relaxation with constant density (rho=1)
    and convergence speed alpha=1.
    
    Args:
        points (np.ndarray): (N, 2) array of point coordinates.
        bounds (tuple): (xmin, xmax, ymin, ymax) defining the bounding box.
		alpha (float): Convergence speed in (0, 1].
        iterations (int): Number of relaxation steps.
        n_padding (int): Number of padding points along each edge.
        
    Returns:
        np.ndarray: Updated point coordinates.
    """
    return modified_lloyd_relaxation(points, bounds, rho=None, alpha=alpha, iterations=iterations, n_padding=n_padding)

def generate_padding_pts(bounds, n_padding, padding_factor=0.1):
    """
    Generates padding points around the bounding box.
    
    Args:
        bounds (tuple): (xmin, xmax, ymin, ymax) defining the bounding box.
        n_padding (int): Number of padding points along each edge.
        padding_factor (float): Factor to determine padding distance relative to max dimension.
        
    Returns:
        np.ndarray: Array of padding point coordinates.
    """
    xmin, xmax, ymin, ymax = bounds
    padding = max(xmax - xmin, ymax - ymin) * padding_factor
    padding_pts_list = []
    
    x_range = np.linspace(xmin - padding, xmax + padding, n_padding)
    y_range = np.linspace(ymin - padding, ymax + padding, n_padding)
    
    # Layer 1
    for x in x_range:
        padding_pts_list.append([x, ymin - padding])
        padding_pts_list.append([x, ymax + padding])
    for y in y_range:
        padding_pts_list.append([xmin - padding, y])
        padding_pts_list.append([xmax + padding, y])

    # Layer 2 (Zigzag)
    dx = x_range[1] - x_range[0]
    dy = y_range[1] - y_range[0]
    padding2 = padding * 2
    for x in x_range + dx / 2:
        padding_pts_list.append([x, ymin - padding2])
        padding_pts_list.append([x, ymax + padding2])
    for y in y_range + dy / 2:
        padding_pts_list.append([xmin - padding2, y])
        padding_pts_list.append([xmax + padding2, y])
    return np.array(padding_pts_list)

def modified_lloyd_relaxation(points, bounds, rho=None, alpha=1.0, iterations=1, samp_pts = 200, n_padding=10, padding_pts=None, return_voronoi=False):
    """
    Performs modified Lloyd relaxation on a set of points with a density function.
    
    The algorithm updates point positions towards the mass centroid of their Voronoi cells,
    weighted by the density function rho.
    
    Args:
        points (np.ndarray): (N, 2) array of point coordinates.
        bounds (tuple): (xmin, xmax, ymin, ymax) defining the bounding box.
        rho (callable, optional): Density function rho(x, y). If None, constant density is assumed.
        alpha (float): Convergence speed in (0, 1].
        iterations (int): Number of relaxation steps.
        samp_pts (int): Number of sampling points for weighted centroid calculation.
        n_padding (int): Number of padding points along each edge.
        return_voronoi (bool): If True, return (new_points, vor).
        
    Returns:
        np.ndarray or tuple: Updated point coordinates, or (updated coordinates, Voronoi object).
    """
    xmin, xmax, ymin, ymax = bounds
    new_points = points.copy()
    
    if padding_pts is None:
        padding_pts = generate_padding_pts(bounds, n_padding)

    vor = None
    for _ in range(iterations):
        # To handle boundary conditions for Voronoi, we can mirror points or use a large bounding box
        # For simplicity in this implementation, we'll use a standard Voronoi and clip/intersect
        # with the bounding box. A common trick is to add "dummy" points far away.
        
        
        pts_to_vor = np.vstack([new_points, padding_pts])
        try:
            vor = Voronoi(pts_to_vor)
        except QhullError:
            # Perturb points slightly to avoid precision errors (e.g., flat simplex)
            perturbation = np.random.normal(0, 1e-10 * max(xmax - xmin, ymax - ymin), pts_to_vor.shape)
            vor = Voronoi(pts_to_vor + perturbation)
        
        updated_pts = []
        for i in range(len(new_points)):
            region_idx = vor.point_region[i]
            region_vertices_indices = vor.regions[region_idx]
            
            if -1 in region_vertices_indices or not region_vertices_indices:
                # This shouldn't happen with dummy points, but safety first
                updated_pts.append(new_points[i])
                continue
                
            vertices = vor.vertices[region_vertices_indices]
            
            # Clip vertices to bounds (simplified: just clamp)
            # For better results, one should use polygon clipping (e.g., Sutherland-Hodgman)
            # but for a general implementation, we'll approximate the centroid.
            
            # Calculate mass centroid c_i
            if rho is None:
                # Ordinary centroid of polygon
                c_i = _calculate_polygon_centroid(vertices)
            else:
                # Weighted centroid using density function rho
                c_i = _calculate_weighted_centroid(vertices, rho, new_points[i], samp_pts)
            
            # Update position: s_i = s_i + alpha * (c_i - s_i)
            new_pos = new_points[i] + alpha * (c_i - new_points[i])
            updated_pts.append(new_pos)
            
        new_points = np.array(updated_pts)
    
    if return_voronoi:
        return new_points, vor
        
    return new_points

def _calculate_polygon_centroid(vertices):
    """Calculates the geometric centroid of a 2D polygon."""
    # Using the formula for centroid of a non-self-intersecting closed polygon
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Close the polygon
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if abs(area) < 1e-10:
        return np.mean(vertices, axis=0)
        
    cx = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6.0 * area)
    cy = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6.0 * area)
    
    return np.array([cx, cy])

def _calculate_weighted_centroid(vertices, rho, point, samples=200):
    """
    Calculates the weighted centroid of a polygon using a density function rho.
    Approximated by sampling within the bounding box of the polygon.
    """
    min_v = np.min(vertices, axis=0)
    max_v = np.max(vertices, axis=0)
    
    # Simple Monte Carlo or Grid sampling within the cell
    # For better performance, one could use a more sophisticated integration
    grid_size = max(2, int(np.sqrt(samples)))
    xs = np.linspace(min_v[0], max_v[0], grid_size)
    ys = np.linspace(min_v[1], max_v[1], grid_size)
    xv, yv = np.meshgrid(xs, ys)
    
    pts = np.vstack([xv.ravel(), yv.ravel()]).T
    
    # Check which points are inside the polygon
    from matplotlib.path import Path
    path = Path(vertices)
    mask = path.contains_points(pts)
    
    inside_pts = pts[mask]
    if len(inside_pts) == 0:
        return _calculate_polygon_centroid(vertices)
        
    weights = rho(inside_pts[:, 0], inside_pts[:, 1])
    total_weight = np.sum(weights)
    
    if total_weight < 1e-10:
        return point # _calculate_polygon_centroid(vertices)
        
    cx = np.sum(inside_pts[:, 0] * weights) / total_weight
    cy = np.sum(inside_pts[:, 1] * weights) / total_weight
    
    return np.array([cx, cy])
