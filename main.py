from rtree import index
from scipy.spatial import KDTree


# Create a set of points in 2D space
points = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

# Construct a k-d tree from the points
tree = KDTree(points)

# Query the tree to find the nearest neighbor to a given point
query_point = (4, 5)
dist, index = tree.query(query_point)

# Print the nearest neighbor
print("Nearest neighbor:", points[index])