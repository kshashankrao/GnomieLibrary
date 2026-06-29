# Inverse Perspective Mapping (IPM)

Functions for computing robust inverse perspective mapping on images, useful for robotics and lane detection.

## Example Usage

```python
import cv2
import numpy as np
from gnomie_library import robust_inverse_perspective_mapping, draw_src_points

image = cv2.imread("image.png")

# Define 4 source points on the perspective image
src_points = np.array([[200, 400], [400, 400], [350, 300], [250, 300]], dtype=np.float32)

# Define 4 destination points for a bird's-eye view
dst_points = np.array([[200, 400], [400, 400], [400, 200], [200, 200]], dtype=np.float32)

output_size = (640, 480)

# Apply IPM
warped_image = robust_inverse_perspective_mapping(image, src_points, dst_points, output_size)

cv2.imshow("Warped", warped_image)
cv2.waitKey(0)
```
