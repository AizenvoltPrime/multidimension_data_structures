import pandas as pd

from pyqtree import Index as QuadTree
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash,MinHashLSH

first_letter = input("Enter first letter: ")
last_letter = input("Enter last letter: ")
awards = int(input("Enter number of awards: "))
sim_threshold = int(input("Enter threshold: "))
sim_threshold /= 100

# Read data from scrapdata.csv
data = pd.read_csv("scrapdata.csv", header=None, names=["surname", "awards", "education"])

# Build a quadtree using surname and awards
le = LabelEncoder()
data['first_letter'] = data['surname'].str[0].str.upper()
X = data[["first_letter", "awards"]].values # Convert to numpy array

le.fit(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
X[:,0] = le.transform(X[:,0]) # Transform first_letter column

bbox = (X[:,0].min(), X[:,1].min(), X[:,0].max(), X[:,1].max())
quadtree = QuadTree(bbox)

for i in range(len(X)):
    x, y = X[i]
    quadtree.insert(i, (x, y, x, y))

def query_quad_tree(range_low, range_high, num_awards):
    low = le.transform([range_low])[0]
    high = le.transform([range_high])[0]
    query_bbox = (low, num_awards+1,high, data['awards'].max())
    matches = quadtree.intersect(query_bbox)
    result = data.iloc[list(matches)]
    return result.iloc[:, :3]

quad_tree_builder = query_quad_tree(first_letter.upper(), last_letter.upper(), awards)

print("The LSH indexes are: ", quad_tree_builder)

# Convert education to vector representation using TF-IDF 
vectorizer = TfidfVectorizer() # Create vectorizer object
Y = vectorizer.fit_transform(quad_tree_builder.iloc[:,2]) # Fit and transform education texts
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
            print("The scientist with the name", data['surname'][quad_tree_builder.index[i]], "is similar with:   ")
        elif i == 1:
            print("The scientist with the name", data['surname'][quad_tree_builder.index[i]], "is similar with:   ")
        elif i == 2:
            print("The scientist with the name", data['surname'][quad_tree_builder.index[i]], "is similar with:   ")
        else:
            print("The scientist with the name", data['surname'][quad_tree_builder.index[i]], "is similar with:   ")
        for j in range(len(final_result[i])):
            print(data['surname'][quad_tree_builder.index[final_result[i][j]]]," \t| " ,data['awards'][quad_tree_builder.index[final_result[i][j]]]," \t| ",data['education'][quad_tree_builder.index[final_result[i][j]]], "\n\n")