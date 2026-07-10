import cv2
import numpy as np

def draw_src_points(image, src_points, color=(0, 0, 255), radius=5, thickness=-1):
    """
    Draws the source points and lines connecting them on the image.

    Args:
        image (numpy.ndarray): The input image.
        src_points (numpy.ndarray): The source points (N x 2).
        color (tuple): The color of the points and lines (BGR format).
        radius (int): The radius of the circles representing the points.
        thickness (int): The thickness of the circles (-1 fills the circles).

    Returns:
        numpy.ndarray: The image with the source points and lines drawn.
    """

    image_with_points = image.copy()
    for i, point in enumerate(src_points):
        x, y = map(int, point)
        cv2.circle(image_with_points, (x, y), radius, color, thickness)

        # Draw lines connecting the points
        if i > 0:
            prev_point = src_points[i - 1]
            prev_x, prev_y = map(int, prev_point)
            cv2.line(image_with_points, (prev_x, prev_y), (x, y), color, 2)

    # Close the loop by connecting the last point to the first
    if len(src_points) > 2:
        first_point = src_points[0]
        last_point = src_points[-1]
        first_x, first_y = map(int, first_point)
        last_x, last_y = map(int, last_point)
        cv2.line(image_with_points, (last_x, last_y), (first_x, first_y), color, 2)

    return image_with_points

def robust_inverse_perspective_mapping(image, src_points, dst_points, output_size,
                                       camera_matrix=None, dist_coeffs=None,
                                       ransac_iterations=1000, ransac_threshold=5.0):
    """Robust inverse perspective mapping with optional undistortion and RANSAC."""
    src_points = np.asarray(src_points)
    dst_points = np.asarray(dst_points)
    
    if src_points.shape != dst_points.shape:
        raise ValueError(f"Source and destination points must have identical shapes, got {src_points.shape} and {dst_points.shape}")
    if src_points.ndim != 2 or src_points.shape[1] != 2:
        raise ValueError(f"Points must be of shape (N, 2), got {src_points.shape}")
    if src_points.shape[0] < 4:
        raise ValueError(f"At least 4 correspondence points are required for perspective mapping, got {src_points.shape[0]}")

    if camera_matrix is not None and dist_coeffs is not None:
        image = cv2.undistort(image, camera_matrix, dist_coeffs)

    src_points_float = src_points.astype(np.float32)
    dst_points_float = dst_points.astype(np.float32)

    M, mask = cv2.findHomography(src_points_float, dst_points_float, cv2.RANSAC, ransac_threshold, maxIters=ransac_iterations)

    if M is None:
        import warnings
        warnings.warn("robust_inverse_perspective_mapping: findHomography failed. Returning a black image.")
        if len(image.shape) == 3:
            return np.zeros((output_size[1], output_size[0], image.shape[2]), dtype=image.dtype)
        return np.zeros((output_size[1], output_size[0]), dtype=image.dtype)

    warped_image = cv2.warpPerspective(image, M, output_size)
    return warped_image

if __name__ == "__main__":
    import os
    # Example Usage:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    kitti_path = os.path.join(project_root, "data/test_data/kitti.png")

    image = cv2.imread(kitti_path)

    if image is None:
        print(f"Error: Could not load image from {kitti_path}")
    else:
        height, width, _ = image.shape

        src_points = np.array([
            [width * 0.4, height * 0.6],
            [width * 0.6, height * 0.6],
            [width * 0.9, height * 0.9],
            [width * 0.1, height * 0.9]
        ])

        dst_points = np.array([
            [0, height],
            [width, height],
            [width, 0],
            [0, 0]
        ])

        output_size = (width, height)

        camera_matrix = None
        dist_coeffs = None

        warped_image = robust_inverse_perspective_mapping(image, src_points, dst_points, output_size,
                                                            camera_matrix, dist_coeffs)

        image_with_points = draw_src_points(image, src_points)

        output_image_path = os.path.join(project_root, "data/test_data/image.png")
        output_ipm_path = os.path.join(project_root, "data/test_data/ipm.png")

        cv2.imwrite(output_image_path, image_with_points)
        cv2.imwrite(output_ipm_path, warped_image)
    
