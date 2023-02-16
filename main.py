import csv
from rtree import index
from scipy.spatial import KDTree

with open('scrapdata.csv') as scrapdata:
    reader = csv.reader(scrapdata)
    data = list(reader)
    
user_input_start_letter = input("Please insert start letter:")

user_input_end_letter = input("Please insert end letter:")

user_input = input("Please insert number of awards:")

names_in_range = []
for i in range(1,len(data)):
    if data[i][0][0] >= user_input_start_letter and data[i][0][0] <=user_input_end_letter:
        names_in_range.append(data[i])
    
threshold_data = []
for i in range(1,len(names_in_range)):
    if int(names_in_range[i][1]) > int(user_input):
        threshold_data.append(names_in_range[i])

print(threshold_data)