import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH

# Define a Node class for the 2D Range Tree
class Node:
    def __init__(self, point):
        self.point = point  # Point in the 2D space (x, y)
        self.left = None  # Left child node
        self.right = None  # Right child node
        self.y_tree = None  # 1D Range Tree for y-coordinates

# Define a RangeTree class for the 2D Range Tree
class RangeTree:
    def __init__(self, points):
        self.root = self.build(points)  # Build the 2D Range Tree

    # Build the 2D Range Tree recursively
    def build(self, points):
        if not points.any():
            return None

        mid = len(points) // 2
        node = Node(points[mid])  # Create a node for the middle point
        node.left = self.build(points[:mid])  # Recursively build the left subtree
        node.right = self.build(points[mid+1:])  # Recursively build the right subtree
        node.y_tree = RangeTree1D(points[:, 1])  # Build the 1D Range Tree for y-coordinates
        return node

# Query the 2D Range Tree for points within a given range
    def query(self, x_min, x_max, y_min=None, y_max=None):
        result = []
        self._query(self.root, x_min, x_max, y_min, y_max, result)  # Start recursive querying
        return result

    # Helper function to query the 2D Range Tree recursively
    def _query(self, node, x_min, x_max, y_min=None, y_max=None, result=None):
        if not node:
            return

        if x_min <= node.point[0] <= x_max:
            # Query the 1D Range Tree for y-coordinates within the given range
            y_result = node.y_tree.query(y_min, y_max)
            for y in y_result:
                result.append((node.point[0], y))  # Add the point to the result

        if node.left and x_min <= node.point[0]:
            self._query(node.left, x_min, x_max, y_min, y_max, result)  # Recurse on the left subtree

        if node.right and x_max >= node.point[0]:
            self._query(node.right, x_min, x_max, y_min, y_max, result)  # Recurse on the right subtree

# Define a Node1D class for the 1D Range Tree
class Node1D:
    def __init__(self, value):
        self.value = value  # Value in the 1D space
        self.left = None  # Left child node
        self.right = None  # Right child node

# Define a RangeTree1D class for the 1D Range Tree
class RangeTree1D:
    def __init__(self, values):
        self.root = self.build(values)  # Build the 1D Range Tree

    # Build the 1D Range Tree recursively
    def build(self, values):
        if not values.any():
            return None

        mid = len(values) // 2
        node = Node1D(values[mid])  # Create a node for the middle value
        node.left = self.build(values[:mid])  # Recursively build the left subtree
        node.right = self.build(values[mid+1:])  # Recursively build the right subtree
        return node

    # Query the 1D Range Tree for values within a given range
    def query(self, min_value=None, max_value=None):
        return self._query(self.root, min_value=min_value, max_value=max_value)  # Start recursive querying

    # Helper function to query the 1D Range Tree recursively
    def _query(self, node, min_value=None, max_value=None):
        if not node:
            return []

        result = []
        if (min_value is None or min_value <= node.value) and (max_value is None or node.value <= max_value):
            result.append(node.value)  # Add the value to the result if it falls within the range

        if node.left and (min_value is None or min_value <= node.value):
            result.extend(self._query(node.left, min_value=min_value, max_value=max_value))  # Recurse on the left subtree

        if node.right and (max_value is None or max_value >= node.value):
            result.extend(self._query(node.right, min_value=min_value, max_value=max_value))  # Recurse on the right subtree
        
        return list(set(result))  # Return the unique values in the result

# Get user input for first letter of surname range, last letter of surname range, minimum number of awards, and similarity threshold for education text.
first_letter = input("Enter first letter: ")
last_letter = input("Enter last letter: ")
awards = int(input("Enter number of awards: "))
sim_threshold = int(input("Enter threshold: "))
sim_threshold /= 100

# Read data from scrapdata.csv containing information about scientists' surnames, number of awards, and education.
data = pd.read_csv("scrapdata.csv", header=None,names=["surname", "awards", "education"])

# Build a range tree using surname and awards.
le = LabelEncoder()
data['first_letter'] = data['surname'].str[0]
X = data[["surname", "awards"]].values.reshape(-1, 2)
X[:, 0] = le.fit_transform(X[:, 0])

tree = RangeTree(X)

# Define a function to query the range tree for scientists within a given surname range and with a minimum number of awards.
def query_range_tree(range_low, range_high, num_awards):
    # Get the first and last letters as integers.
    first_letter = ord(range_low[0].upper())
    last_letter = ord(range_high[0].upper())
    
    # Query the tree with the given range and number of awards.
    result = tree.query(first_letter, last_letter, num_awards)
    
    # Convert the result to a DataFrame.
    result = pd.DataFrame(result, columns=['surname', 'awards'])
    
    # Decode the surname values.
    result['surname'] = le.inverse_transform(result['surname'])
    
    # Merge with the original data to get the education column.
    result = pd.merge(result, data[['surname', 'education']], on='surname')
    
    return result

lsh_builder = query_range_tree(first_letter, last_letter, awards)

print("The LSH indexes are: \n", lsh_builder)

# Convert education to vector representation using TF-IDF.
vectorizer = TfidfVectorizer() # Create vectorizer object.
Y = vectorizer.fit_transform(lsh_builder.iloc[:,2]) # Fit and transform education texts.

# Apply MinHash on vectors to create hash signatures.
lsh = MinHashLSH(threshold=sim_threshold) # Create MinHashLSH object.
for i in range(Y.shape[0]): # Loop over each vector.
    mh = MinHash(num_perm=128) # Create MinHash object with 10 permutations (you can change this).
    for j in Y[i].nonzero()[1]: # Loop over each non-zero element in vector.
        mh.update(str(j).encode('utf8')) # Update MinHash with element value encoded as bytes.
        lsh.insert(i, mh,check_duplication=False) # Insert index and MinHash into LSHs

# Define a function to query LSH for groups of scientists with similar education based on a user-defined similarity threshold.
def query_lsh(matrix):
    results = []
    for i in range(matrix.shape[0]):
        vector = matrix[i]
        mh_query = MinHash(num_perm=128) # Create MinHash object for query vector. 
        for j in vector.nonzero()[1]: # Loop over each non-zero element in vector. 
            mh_query.update(str(j).encode('utf8')) # Update MinHash with element value encoded as bytes. 
            result = lsh.query(mh_query) # Query LSH with query MinHash and get result as a list of indices. 
        results.append(result)
    return results 

final_result = query_lsh(Y)
print("The groups of similarities are: ", final_result,"\n\n\n\n")

count = 0
for i in range(len(final_result)):
    if len(final_result[i]) > 1:
        print("\n\n")
    if i == 0:
        print("The scientist with the name", data['surname'][lsh_builder.index[i]], "is similar with: ")
    elif i == 1:
        print("The scientist with the name", data['surname'][lsh_builder.index[i]], "is similar with: ")
    elif i == 2:
        print("The scientist with the name", data['surname'][lsh_builder.index[i]], "is similar with: ")
    else:
        print("The scientist with the name", data['surname'][lsh_builder.index[i]], "is similar with: ")
    for j in range(len(final_result[i])):
        print(data['surname'][lsh_builder.index[final_result[i][j]]]," \t| " ,data['awards'][lsh_builder.index[final_result[i][j]]]," \t| ",data['education'][lsh_builder.index[final_result[i][j]]], "\n\n")