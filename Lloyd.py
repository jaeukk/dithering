import numpy as np
from scipy.spatial import Voronoi

def lloyd_relaxation(points, bounds, alpha=0.2, iterations=1, padding = None):
    """
    Performs ordinary Lloyd relaxation on a set of points.
    
    This is a special case of modified Lloyd relaxation with constant density (rho=1)
    and convergence speed alpha=1.
    
    Args:
        points (np.ndarray): (N, 2) array of point coordinates.
        bounds (tuple): (xmin, xmax, ymin, ymax) defining the bounding box.
		alpha (float): Convergence speed in (0, 1].
        iterations (int): Number of relaxation steps.
        
    Returns:
        np.ndarray: Updated point coordinates.
    """
    return modified_lloyd_relaxation(points, bounds, rho=None, alpha=alpha, iterations=iterations, padding_pts=padding)

def modified_lloyd_relaxation(points, bounds, rho=None, alpha=1.0, iterations=1, samp_pts = 200, padding_pts=None):
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
        
    Returns:
        np.ndarray: Updated point coordinates.
    """
    xmin, xmax, ymin, ymax = bounds
    new_points = points.copy()
    
    # Add dummy points to ensure all real points have closed Voronoi cells within bounds
    # padding = max(xmax - xmin, ymax - ymin) * 2
    # dummy_points = np.array([
    #     [xmin - padding, ymin - padding],
    #     [xmin - padding, ymax + padding],
    #     [xmax + padding, ymin - padding],
    #     [xmax + padding, ymax + padding]
    # ])
    padding = max(xmax - xmin, ymax - ymin) * 0.1
    padding_pts = np.zeros((0,2))
    
    x_range = np.linspace(xmin - padding, xmax + padding, 10)
    y_range = np.linspace(ymin - padding, ymax + padding, 10)
    
    for x in x_range:
        padding_pts = np.vstack([padding_pts, [x, ymin - padding]])
        padding_pts = np.vstack([padding_pts, [x, ymax + padding]])
    for y in y_range:
        padding_pts = np.vstack([padding_pts, [xmin - padding, y]])
        padding_pts = np.vstack([padding_pts, [xmax + padding, y]])

    for _ in range(iterations):
        # To handle boundary conditions for Voronoi, we can mirror points or use a large bounding box
        # For simplicity in this implementation, we'll use a standard Voronoi and clip/intersect
        # with the bounding box. A common trick is to add "dummy" points far away.
        
        
        pts_to_vor = np.vstack([new_points, padding_pts])
        vor = Voronoi(pts_to_vor)
        
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
                c_i = _calculate_weighted_centroid(vertices, rho, samp_pts)
            
            # Update position: s_i = s_i + alpha * (c_i - s_i)
            new_pos = new_points[i] + alpha * (c_i - new_points[i])
            updated_pts.append(new_pos)
            
        new_points = np.array(updated_pts)
        
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

def _calculate_weighted_centroid(vertices, rho, samples=200):
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
        return _calculate_polygon_centroid(vertices)
        
    cx = np.sum(inside_pts[:, 0] * weights) / total_weight
    cy = np.sum(inside_pts[:, 1] * weights) / total_weight
    
    return np.array([cx, cy])
