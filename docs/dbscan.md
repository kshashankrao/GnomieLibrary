# DBSCAN

Density-Based Spatial Clustering of Applications with Noise.

## Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from gnomie_library import DBSCAN

# Generate sample data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Initialize DBSCAN with epsilon=0.2 and min_points=5
db = DBSCAN(eps=0.2, min_pts=5)
db.process(X)

# Retrieve clusters
clusters = db.get_clusters()

# Example visualization
for cluster in clusters:
    points = cluster.get_points_np()
    if points.size > 0:
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster.label}')
        
plt.show()
```
