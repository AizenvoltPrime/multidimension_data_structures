# Import libraries
import pandas as pd
import numpy as np

# Read data from scrapdata.csv
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHashLSHForest, MinHash,MinHashLSH

# Read data from scrapdata.csv
data = pd.read_csv("scrapdata.csv", header=None, names=["surname", "awards", "education"])

# Build a k-d tree using surname and awards
le = LabelEncoder()
data['first_letter'] = data['surname'].str[0]
X = data[["surname", "awards"]].values # Convert to numpy array
X[:,0] = le.fit_transform(X[:,0]) # Transform surname column
tree = KDTree(X) # Create k-d tree object

def query_kd_tree(range_low, range_high, num_awards):
    low = ord(range_low[0].upper())
    ind = tree.query([[low,num_awards]], k=len(X), return_distance=False)[0]
    result = data.iloc[ind]
    result = result[result['first_letter'] >= range_low[0].upper()]
    result = result[result['first_letter'] <= range_high[0].upper()]
    result = result[result['awards'] >= num_awards]
    return result.iloc[:, :3]

# Convert education to vector representation using TF-IDF 
vectorizer = TfidfVectorizer() # Create vectorizer object
Y = vectorizer.fit_transform(data["education"]) # Fit and transform education texts
# Apply MinHash on vectors to create hash signatures 
lsh = MinHashLSH(threshold=0.4) # Create MinHashLSH object with similarity threshold (you can change this)
for i in range(Y.shape[0]): # Loop over each vector 
    mh = MinHash(num_perm=128) # Create MinHash object with 10 permutations (you can change this)
    for j in Y[i].nonzero()[1]: # Loop over each non-zero element in vector 
        mh.update(str(j).encode('utf8')) # Update MinHash with element value encoded as bytes 
        lsh.insert(i, mh, check_duplication=False) # Insert index and MinHash into LSH

# Define a function that returns points based on query vector and similarity percentage (not needed anymore)
def query_lsh(vector):
    mh_query = MinHash(num_perm=128) # Create MinHash object for query vector 
    for j in vector.nonzero()[1]: # Loop over each non-zero element in vector 
        mh_query.update(str(j).encode('utf8')) # Update MinHash with element value encoded as bytes 
        result = lsh.query(mh_query) # Query LSH with query MinHash and get result as a list of indices
    return result 

# Combine both functions to answer queries of the form: "Find computer scientists from scrapdata.csv with >60% education similarity, whose letter is in the interval [A, G], and who have won > 4 awards."
def query_combined(range_low,range_high,threshold):
    result1 = query_kd_tree(range_low,range_high, threshold) # Query k-d tree first 
    result2 = query_lsh(Y[result1.index]) # Query LSH on the subset of result1  
    final_result = []
    print(result1)
    print(result2)
    for i in range(result1.shape[0]):
        if result1.iloc[i,1:1].name in result2:
            final_result.append(result1.iloc[i].tolist())
    return final_result # Return final result as a pandas dataframe 

# Test with an example query
test = query_combined("K","O", 10) # Remove 4 from the arguments

for i in test:
    print(i,"\n\n")
    
#[67, 68, 102, 70, 44, 47, 48, 52, 85]