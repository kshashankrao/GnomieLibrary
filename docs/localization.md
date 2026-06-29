# Localization

Particle filters and base filters for 2D map localization.

## Example Usage

```python
import cv2
from gnomie_library import ParticleFilter

# Load map image
map_img = cv2.imread("map.png", 0)

if map_img is not None:
    # Initialize the particle filter
    pf = ParticleFilter(map_img, num_particles=3000)

    # Predict step (moving the robot)
    fwd, turn = 5.0, 0.1
    pf.predict(fwd, turn)

    # Update step (resampling based on sensors)
    pf.update()

    # Get the best estimated position
    print("Estimated Position:", pf.estimate_state())
```
