import csv
import mmh3
import nltk
nltk.download('stopwords')
from rtree import index
from scipy.spatial import KDTree
from nltk.corpus import stopwords

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
for i in range(1,len(names_in_range)):
    if int(names_in_range[i][1]) > int(user_input_awards):
        threshold_data.append(names_in_range[i])
        

# Define stop words to remove from the education field
stop_words = set(stopwords.words('english'))

# Tokenize and preprocess the education field
def preprocess_education(threshold_data):
    threshold_data = threshold_data.lower() # Convert to lowercase
    education_tokens = threshold_data.split() # Tokenize
    education_tokens = [token for token in education_tokens if token not in stop_words] # Remove stop words
    return education_tokens

# Define the number of hash functions and the length of the feature vector
num_hashes = 5
feature_len = 10

# Define the hashing function
def hash_token(token, seed):
    hash_val = mmh3.hash(token, seed)
    bit_string = bin(hash_val)  # remove the '0b' prefix from the binary string
    bit_string = bit_string[3:]
    feature_vector = [int(bit) for bit in bit_string]
    feature_vector = feature_vector[-feature_len:]
    feature_vector += [0] * (feature_len - len(feature_vector))
    return feature_vector

# Define the feature vector for an education field
def education_feature_vector(threshold_data):
    feature_vector = [0] * feature_len
    for i in range(num_hashes):
        tokens=str(preprocess_education(threshold_data))
        token_feature = hash_token(tokens, i)
        for j in range(len(token_feature)):
            feature_vector[j] += token_feature[j]
    return feature_vector

for i in range(len(threshold_data)):
    hash_vector = education_feature_vector(threshold_data[i][2])
    print(hash_vector)