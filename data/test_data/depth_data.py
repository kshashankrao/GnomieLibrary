import numpy as np
import cv2
def create_dog_depth_map(height, width):
    """Creates a simplified depth map of a dog with gradient effects."""
    depth_map = np.zeros((height, width), dtype=np.uint8)

    # Head (gradient from center)
    head_center = (width // 2, height // 3)
    head_radius_x, head_radius_y = width // 6, height // 8
    for i in range(height):
        for j in range(width):
            if ((j - head_center[0])**2 / head_radius_x**2 + (i - head_center[1])**2 / head_radius_y**2) <= 1:
                dist_to_center = np.sqrt((j - head_center[0])**2 + (i - head_center[1])**2)
                depth_map[i, j] = int(255 * (1 - dist_to_center / max(head_radius_x, head_radius_y)))

    # Snout (gradient)
    snout_center = (width // 2, height // 4)
    snout_radius_x, snout_radius_y = width // 12, height // 16
    for i in range(height):
        for j in range(width):
            if ((j - snout_center[0])**2 / snout_radius_x**2 + (i - snout_center[1])**2 / snout_radius_y**2) <= 1:
                dist_to_center = np.sqrt((j - snout_center[0])**2 + (i - snout_center[1])**2)
                depth_map[i, j] = max(0, int(200 * (1 - dist_to_center / max(snout_radius_x, snout_radius_y))))

    # Body (gradient)
    body_center = (width // 2, height // 2)
    body_radius_x, body_radius_y = width // 4, height // 3
    for i in range(height):
        for j in range(width):
            if ((j - body_center[0])**2 / body_radius_x**2 + (i - body_center[1])**2 / body_radius_y**2) <= 1:
                dist_to_center = np.sqrt((j - body_center[0])**2 + (i - body_center[1])**2)
                depth_map[i, j] = max(0, int(150 * (1 - dist_to_center / max(body_radius_x, body_radius_y))))

    # Legs (constant depth)
    cv2.rectangle(depth_map, (width // 3, height * 3 // 4), (width // 3 + width // 10, height - 1), 100, -1)
    cv2.rectangle(depth_map, (width * 2 // 3, height * 3 // 4), (width * 2 // 3 + width // 10, height - 1), 100, -1)

    # Tail (gradient)
    tail_center = (width * 4 // 5, height // 2)
    tail_radius_x, tail_radius_y = width // 12, height // 12
    for i in range(height):
        for j in range(width):
            if ((j - tail_center[0])**2 / tail_radius_x**2 + (i - tail_center[1])**2 / tail_radius_y**2) <= 1:
                dist_to_center = np.sqrt((j - tail_center[0])**2 + (i - tail_center[1])**2)
                depth_map[i, j] = max(0, int(120 * (1 - dist_to_center / max(tail_radius_x, tail_radius_y))))

    return depth_map