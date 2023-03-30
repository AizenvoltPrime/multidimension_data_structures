import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH

class Node:
    def __init__(self, point):
        self.point = point
        self.left = None
        self.right = None
        self.y_tree = None

class RangeTree:
    def __init__(self, points):
        self.root = self.build(points)

    def build(self, points):
        if not points.any():
            return None

        mid = len(points) // 2
        node = Node(points[mid])
        node.left = self.build(points[:mid])
        node.right = self.build(points[mid+1:])
        node.y_tree = RangeTree1D(points[:, 1])
        return node

    def query(self, x_min, x_max, y_min=None, y_max=None):
        return self._query(self.root, x_min, x_max, y_min, y_max)

    def _query(self, node, x_min, x_max, y_min=None, y_max=None):
        if not node:
            return []

        result = []
        if x_min <= node.point[0] <= x_max:
            result.extend(node.y_tree.query(y_min, y_max))

        if node.left and x_min <= node.point[0]:
            result.extend(self._query(node.left, x_min, x_max, y_min, y_max))

        if node.right and x_max >= node.point[0]:
            result.extend(self._query(node.right, x_min, x_max, y_min, y_max))

        return result

class Node1D:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class RangeTree1D:
    def __init__(self, values):
        self.root = self.build(values)

    def build(self, values):
        if not values.any():
            return None

        mid = len(values) // 2
        node = Node1D(values[mid])
        node.left = self.build(values[:mid])
        node.right = self.build(values[mid+1:])
        return node

    def query(self, min_value=None, max_value=None):
        return self._query(self.root,min_value=min_value,max_value=max_value)

    def _query(self,node,min_value=None,max_value=None):
        if not node:
            return []

        result = []
        if (min_value is None or min_value <= node.value) and (max_value is None or node.value <= max_value):
            result.append(node.value)

        if node.left and (min_value is None or min_value <= node.value):
            result.extend(self._query(node.left,min_value=min_value,max_value=max_value))

        if node.right and (max_value is None or max_value >= node.value):
            result.extend(self._query(node.right,min_value=min_value,max_value=max_value))
        return list(set(result))

first_letter = input("Enter first letter: ")
last_letter = input("Enter last letter: ")
awards = int(input("Enter number of awards: "))
sim_threshold = int(input("Enter threshold: "))
sim_threshold /= 100

# Read data from scrapdata.csv
data = pd.read_csv("scrapdata.csv", header=None,names=["surname", "awards", "education"])

# Build a range tree using surname and awards
le = LabelEncoder()
data['first_letter'] = data['surname'].str[0]
X = data[["surname", "awards"]].values.reshape(-1, 2)
X[:, 0] = le.fit_transform(X[:, 0])

tree = RangeTree(X)


def query_range_tree(range_low, range_high, num_awards):
    mask = (data['first_letter'] >= range_low[0].upper()) & (data['first_letter'] <= range_high[0].upper()) & (data['awards'] > num_awards)
    result = data[mask]
    return result.iloc[:, :3]


lsh_builder = query_range_tree(first_letter, last_letter, awards)

print("The LSH indexes are: ", lsh_builder)

# Convert education to vector representation using TF-IDF
vectorizer = TfidfVectorizer()  # Create vectorizer object
Y = vectorizer.fit_transform(lsh_builder.iloc[:, 2])  # Fit and transform education texts
# Apply MinHash on vectors to create hash signatures
lsh = MinHashLSH(threshold=sim_threshold)  # Create MinHashLSH object
for i in range(Y.shape[0]):  # Loop over each vector
    mh = MinHash(num_perm=128)  # Create MinHash object with 10 permutations (you can change this)
    for j in Y[i].nonzero()[1]:  # Loop over each non-zero element in vector
        mh.update(str(j).encode('utf8'))  # Update MinHash with element value encoded as bytes
        lsh.insert(i, mh,check_duplication=False)  # Insert index and MinHash into LSH


def query_lsh(matrix):
    results = []
    for i in range(matrix.shape[0]):
        vector = matrix[i]
        mh_query = MinHash(num_perm=128) # Create MinHash object for query vector 
        for j in vector.nonzero()[1]: # Loop over each non-zero element in vector 
            mh_query.update(str(j).encode('utf8')) # Update MinHash with element value encoded as bytes 
            result = lsh.query(mh_query) # Query LSH with query MinHash and get result as a list of indices
        results.append(result)
    return results 

final_result = query_lsh(Y)
print("The groups of similarities are: ", final_result, "\n\n\n\n")

count = 0
for i in range(len(final_result)):
    if len(final_result[i]) > 1:
        print("\n\n")
        if i == 0:
            print("The scientist with the name", data['surname'][lsh_builder.index[i]], "is similar with:   ")
        elif i == 1:
            print("The scientist with the name", data['surname'][lsh_builder.index[i]], "is similar with:   ")
        elif i == 2:
            print("The scientist with the name", data['surname'][lsh_builder.index[i]], "is similar with:   ")
        else:
            print("The scientist with the name", data['surname'][lsh_builder.index[i]], "is similar with:   ")
        for j in range(len(final_result[i])):
            print(data['surname'][lsh_builder.index[final_result[i][j]]]," \t| " ,data['awards'][lsh_builder.index[final_result[i][j]]]," \t| ",data['education'][lsh_builder.index[final_result[i][j]]], "\n\n")