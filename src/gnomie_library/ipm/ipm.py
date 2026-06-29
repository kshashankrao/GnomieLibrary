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

    if camera_matrix is not None and dist_coeffs is not None:
        image = cv2.undistort(image, camera_matrix, dist_coeffs)

    src_points_float = src_points.astype(np.float32)
    dst_points_float = dst_points.astype(np.float32)

    M, mask = cv2.findHomography(src_points_float, dst_points_float, cv2.RANSAC, ransac_threshold, maxIters=ransac_iterations)

    if M is None:
        print("Error: findHomography failed. Returning original image.")
        return image

    warped_image = cv2.warpPerspective(image, M, output_size)
    return warped_image

# Example Usage:
image = cv2.imread("/mnt/d/DeepLearning/GnomieLibrary/data/test_data/kitti.png")  # Replace with your image path

if image is None:
    print("Error: Could not load image.")
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

    cv2.imwrite("image.png", image_with_points)
    cv2.imwrite("ipm.png", warped_image)
    
