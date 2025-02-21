import numpy as np
from sfm import EightPointAlgorithm

# Example points and normalization factor
pts1 = np.array([[100, 200], [300, 400], [500, 600], [700, 800]])
pts2 = np.array([[110, 210], [310, 410], [510, 610], [710, 810]])
M = 1000  # Example normalization factor

# Create an instance of the class
eight_point = EightPointAlgorithm()

# Compute the fundamental matrix
F = eight_point(pts1, pts2, M)

print("Fundamental Matrix:")
print(F)