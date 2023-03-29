import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash,MinHashLSH

class Node:
    def __init__(self, point):
        self.point = point
        self.left = None
        self.right = None

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
        return node

    def query(self, x_min, x_max, y_min, y_max):
        return self._query(self.root, x_min, x_max, y_min, y_max)

    def _query(self, node, x_min, x_max, y_min, y_max):
        if not node:
            return []

        if x_min <= node.point[0] <= x_max and y_min <= node.point[1] <= y_max:
            return [node.point]

        if node.left and x_min <= node.point[0]:
            result = self._query(node.left, x_min, x_max, y_min, y_max)
        else:
            result = []

        if node.right and x_max >= node.point[0]:
            result += self._query(node.right, x_min, x_max, y_min, y_max)

        return result

first_letter = input("Enter first letter: ") 
last_letter = input("Enter last letter: ") 
awards = int(input("Enter number of awards: ")) 
sim_threshold = int(input("Enter threshold: ")) 
sim_threshold /= 100 

# Read data from scrapdata.csv 
data = pd.read_csv("scrapdata.csv", header=None, names=["surname", "awards", "education"]) 

# Build a range tree using surname and awards 
le = LabelEncoder() 
data['first_letter'] = data['surname'].str[0] 
X = data[["surname", "awards"]].values 
X[:,0] = le.fit_transform(X[:,0]) 
tree = RangeTree(X) 

def query_range_tree(range_low, range_high, num_awards): 
    low = ord(range_low[0].upper()) 
    result = tree.query(low-1, ord(range_high[0].upper()), num_awards-1,num_awards) 
    result_df = pd.DataFrame(result) 
    result_df.columns=["surname", "awards"] 
    result_df["education"] = data.loc[result_df.index]["education"].values 
    result_df["first_letter"] = data.loc[result_df.index]["first_letter"].values
    return result_df

lsh_builder = query_range_tree(first_letter, last_letter , awards) 

print(lsh_builder)
for i in range(len(lsh_builder)): 
    print(lsh_builder.iloc[i]['surname'], "LSH \n\n\n\n\n")
    # for j in range(len(data)):
        # print(data.index[j], "DATA \n\n\n\n\n")
        # if data.index[j] == lsh_builder.iloc[i]['surname']:
            # print(data.iloc[lsh_builder.iloc[i]['surname']]['surname'], "\n\n\n")
            # print(data.iloc[j]['awards'])
            # print(data.iloc[j]['education'])
#print("The LSH indexes are: ", lsh_builder, "\n\n\n\n")

# Convert education to vector representation using TF-IDF 
vectorizer = TfidfVectorizer() # Create vectorizer object
Y = vectorizer.fit_transform(lsh_builder.iloc[:,2]) # Fit and transform education texts
# Apply MinHash on vectors to create hash signatures 
lsh = MinHashLSH(threshold = sim_threshold) # Create MinHashLSH object
for i in range(Y.shape[0]): # Loop over each vector 
    mh = MinHash(num_perm=128) # Create MinHash object with 10 permutations (you can change this)
    for j in Y[i].nonzero()[1]: # Loop over each non-zero element in vector 
        mh.update(str(j).encode('utf8')) # Update MinHash with element value encoded as bytes 
        lsh.insert(i, mh, check_duplication=False) # Insert index and MinHash into LSH

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