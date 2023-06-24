import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH

# Get user inputs for first letter, last letter, number of awards and similarity threshold
first_letter = input("Enter first letter: ")
last_letter = input("Enter last letter: ")
awards = int(input("Enter number of awards: "))
sim_threshold = int(input("Enter threshold: "))
sim_threshold /= 100

# Read data from scrapdata.csv
data = pd.read_csv("scrapdata.csv", header=None, names=["surname", "awards", "education"])

# Build a KDTree using surname and awards
le = LabelEncoder()
data['first_letter'] = data['surname'].str[0].str.upper()
X = data[["first_letter", "awards"]].values # Convert to numpy array

le.fit(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
X[:,0] = le.transform(X[:,0]) # Transform first_letter column

kdtree = KDTree(X)

def query_kd_tree(range_low, range_high, num_awards):
    # Query the KDTree to find surnames within the given range and with more than the given number of awards
    low = le.transform([range_low])[0]
    high = le.transform([range_high])[0]
    query_kd = (low, num_awards+1)
    matches = kdtree.query_radius([query_kd], r=high-low)[0]
    result = data.iloc[list(matches)]
    result = result[result['awards'] > num_awards] # Filter out rows that do not meet the specified number of awards
    result = result.sort_index() # Sort the resulting DataFrame by its index
    return result.iloc[:, :3]

kd_tree_builder = query_kd_tree(first_letter.upper(), last_letter.upper(), awards)

print("The KD-tree query results are: \n", kd_tree_builder)

# Convert education to vector representation using TF-IDF 
vectorizer = TfidfVectorizer() # Create vectorizer object
Y = vectorizer.fit_transform(kd_tree_builder.iloc[:,2]) # Fit and transform education texts

# Apply MinHash on vectors to create hash signatures 
lsh = MinHashLSH(threshold=sim_threshold) # Create MinHashLSH object
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
            print("The scientist with the name", data['surname'][kd_tree_builder.index[i]], "is similar with: ")
        elif i == 1:
            print("The scientist with the name", data['surname'][kd_tree_builder.index[i]], "is similar with: ")
        elif i == 2:
            print("The scientist with the name", data['surname'][kd_tree_builder.index[i]], "is similar with: ")
        else:
            print("The scientist with the name", data['surname'][kd_tree_builder.index[i]], "is similar with: ")
        for j in range(len(final_result[i])):
            print(data['surname'][kd_tree_builder.index[final_result[i][j]]]," \t| " ,data['awards'][kd_tree_builder.index[final_result[i][j]]]," \t| ",data['education'][kd_tree_builder.index[final_result[i][j]]], "\n\n")