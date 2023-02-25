import csv
import nltk
import string
nltk.download('stopwords')
from scipy.spatial import KDTree
from nltk.corpus import stopwords
from datasketch import MinHashLSHForest, MinHash

with open('scrapdata.csv') as scrapdata:
    reader = csv.reader(scrapdata)
    data = list(reader)
    
user_input_start_letter = input("Please insert start letter:")

user_input_end_letter = input("Please insert end letter:")

user_input_awards = int(input("Please insert number of awards:"))

names_in_range = []
for i in range(1,len(data)):
    if data[i][0][0] >= user_input_start_letter and data[i][0][0] <=user_input_end_letter:
        names_in_range.append(data[i])
    
threshold_data = []
for i in range(0,len(names_in_range)):
    if int(names_in_range[i][1]) >= int(user_input_awards):
        threshold_data.append(names_in_range[i])
        

# Define stop words to remove from the education field
stop_words = set(stopwords.words('english'))

# Tokenize and preprocess the education field
def preprocess_education(threshold_data):
    threshold_data = threshold_data.lower() # Convert to lowercase
    threshold_data = threshold_data.translate(str.maketrans('', '', string.punctuation))
    education_tokens = threshold_data.split() # Tokenize
    shingle_tokens = []
    for token in education_tokens:
        if token not in stop_words:
            shingle_tokens.append(token)
    return shingle_tokens

# Define the feature vector for an education field
def education_feature_vector(education):
    shingles = preprocess_education(education)
    m = MinHash(num_perm=128)
    for shingle in shingles:
        m.update(shingle.encode('utf-8'))
    return m

# Create LSH forest and implement KD-Tree
num_perm = 128
n_neighbors = 5
feature_vectors = []
lsh_forest = MinHashLSHForest(num_perm=num_perm)
for i in range(len(threshold_data)):
    m = education_feature_vector(threshold_data[i][2])
    lsh_forest.add(str(i), m)
    feature_vectors.append(list(m.hashvalues))

# Index the LSH forest
lsh_forest.index()

# Query the LSH forest to find similar education fields
query_education = threshold_data[4][2]
query_m = education_feature_vector(query_education)
result = lsh_forest.query(query_m, n_neighbors)
similar_indices_lsh = [int(idx) for idx in result]

#Create KD-Tree    
kd_tree = KDTree(feature_vectors)

# Query the kd-tree to find similar education fields
distances, indices = kd_tree.query([list(query_m.hashvalues)], k=n_neighbors)
similar_indices_kd = indices[0].tolist()

# Combine the results from LSH and kd-tree
similar_indices_combined = list(set(similar_indices_lsh) | set(similar_indices_kd))

# # # Print the original and similar education fields
# print(query_education, "\n")
# print("These are the LSH most similar Educations: \n\n")
# for val in similar_indices_lsh:
#     print(threshold_data[val][2], "\n")
    
# print("These are the KD most similar Educations: \n\n")
# for val in similar_indices_kd:
#     print(threshold_data[val][2], "\n")

print("This is the union of the most similar Educations: \n\n")
for val in similar_indices_combined:
    print(threshold_data[val][2], "\n")