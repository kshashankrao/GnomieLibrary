"""
The dbscan has 3 main points.
1. Core point - The point with atleast min_pts points within eps distance
2. Border point - The points with points within eps distance but less than min_pts
3. Noise point -  The points with no points within eps distance

For each core point:
1. the neighbors are found 
2. each neighbor is classified into core, border and noise.
3. For each of the core point, step 1 and 2 is repeated until all the points are classified.
"""

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.visited = False

class Point_Utils:
  @staticmethod
  def convert_np_pts(pts):
    return [Point(pt[0], pt[1]) for pt in pts]

  @staticmethod
  def convert_pts_to_np(pts):
    return np.array([[pt.x, pt.y] for pt in pts])

  @staticmethod
  def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)

class Cluster:
  def __init__(self, label):
    self.pts = np.empty((0))
    self.label = label
  
  def update_points(self, pt):
    self.pts = np.append(self.pts, pt)

  def get_points(self):
    return self.pts
  
  def get_points_np(self):
    return Point_Utils.convert_pts_to_np(self.pts)
  
class DBSCAN:
  def __init__(self, eps, min_pts):
    self.eps = eps
    self.min_pts = min_pts
    self.clusters = []
    self.label = 1

  def create_cluster(self, label):
    cluster = Cluster(label)
    self.clusters.append(cluster)
    return cluster
  
  def get_clusters(self):
    return self.clusters

  def get_neighbors(self,pt, pts):
    neighbors = []
    
    for i, pt_n in enumerate(pts):
      dist = Point_Utils.euclidean_distance(pt, pt_n)
      if dist <= self.eps and dist > 0:
        neighbors.append(pt_n)

    return neighbors
    
  def process(self, pts):
    noise_cluster = self.create_cluster(0)
    pts = Point_Utils.convert_np_pts(pts)

    for i, pt in enumerate(pts):
      if pt.visited:
        continue

      pt.visited = True
      neighbors = self.get_neighbors(pt, pts)
      
      # Core point
      if len(neighbors) >= self.min_pts:
        cluster = self.create_cluster(self.label)
        self.label += 1
        cluster.update_points(pt)

        '''
        Keep on the adding the core point's neighbors of neighbors of neighbors 
        ... to the original neighbors array of core point.
        Expand only for core points, and directly add the border point to the
        cluster.
        '''
        while neighbors:
          pt_n = neighbors.pop()
          if pt_n.visited:
            continue
          pt_n.visited = True
          neighbors_n = self.get_neighbors(pt_n, pts)
          
          # Update the neighbors of the core point
          if len(neighbors_n) >= self.min_pts:
            neighbors.extend(neighbors_n)
            cluster.update_points(pt_n)
          
          # Border point
          else:
            cluster.update_points(pt_n)

      else:
        noise_cluster.update_points(pt)
        
if __name__ == "__main__":
  X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

  db = DBSCAN(0.2, 5)
  db.process(X)

  clusters = db.get_clusters()
  plt.figure(figsize=(16, 6))

  plt.subplot(1, 2, 1)
  plt.scatter(X[:, 0], X[:, 1])
  plt.title('Original Data')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.subplot(1, 2, 2)

  colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  for i, cluster in enumerate(clusters):
      points = cluster.get_points_np()
      if points.size > 0:
          plt.scatter(points[:, 0], points[:, 1], color=colors[i % len(colors)], label=f'Cluster {cluster.label}')

  plt.title('DBSCAN Clustering Results')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.legend()
  plt.tight_layout() 
  plt.show()
