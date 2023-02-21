import csv
import mmh3
import nltk
import string
nltk.download('stopwords')
from rtree import index
from scipy.spatial import KDTree
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

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

# Define the number of hash functions and the length of the feature vector
num_hashes = 10
feature_len = 20

# Define the hashing function
def hash_token(shingle, seed):
    hash_val = mmh3.hash(shingle, seed)
    bit_string = bin(hash_val)  # remove the '0b' prefix from the binary string
    bit_string = bit_string[3:]
    feature_vector = [int(bit) for bit in bit_string]
    feature_vector = feature_vector[-feature_len:]
    feature_vector += [0] * (feature_len - len(feature_vector))
    return feature_vector

# Define the feature vector for an education field
def education_feature_vector(threshold_data):
    feature_vector = [0] * feature_len
    education_tokens = preprocess_education(threshold_data)
    for i in range(len(education_tokens) - 2):
        shingle = education_tokens[i] + " " + education_tokens[i + 1] + " " + education_tokens[i + 2] 
        for j in range(num_hashes):
            token_feature = hash_token(shingle, j)
            for k in range(len(token_feature)):
                feature_vector[k] += token_feature[k]
    return feature_vector

for i in range(len(threshold_data)):
    hash_vector = education_feature_vector(threshold_data[i][2])

n_neighbors = 5
nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')

education_vectors=[]
for row in threshold_data:
    education_vectors.append(education_feature_vector(row[2]))
nn.fit(education_vectors)

# Query the NearestNeighbors model to find similar education fields
query_education = threshold_data[4][2]
query_vector = education_feature_vector(query_education)
similar_indices = nn.kneighbors([query_vector], n_neighbors=5, return_distance=False)[0]

print(threshold_data[4][2],"\n\n")
for val in similar_indices:
    print(threshold_data[val][2],"\n\n\n")  