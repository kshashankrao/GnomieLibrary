# Structure from Motion (SfM)

Implementation of the Eight-Point Algorithm for fundamental matrix estimation.

## Example Usage

```python
import numpy as np
from gnomie_library import EightPointAlgorithm

# Dummy matched points
pts1 = np.random.rand(10, 2) * 100
pts2 = np.random.rand(10, 2) * 100
M = 100.0 # Normalization factor

# Estimate fundamental matrix
sfm = EightPointAlgorithm()
F = sfm(pts1, pts2, M)

print("Fundamental Matrix:")
print(F)
```
