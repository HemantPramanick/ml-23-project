# -*- coding: utf-8 -*-
"""pincode_ml_project_phase_2ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O49rQbBEWPRHVUe47JJgMR57e6GChSlv
"""

from geopy.geocoders import Nominatim

# Create a geocoder instance
geolocator = Nominatim(user_agent="my_geocoder")

# Define the longitude and latitude coordinates for the flats
coordinates = [(18.5438, 73.9438), (24.69028, 78.41889), (19.216097, 72.821721), (24.69028, 78.41889), (20.448627, 85.897743), (12.96991, 77.59796), (26.864668, 81.0654), (30.662283, 76.822397), (12.96991, 77.59796), (30.752321, 76.772816), (18.996549, 72.842967), (22.461348, 88.318428), (16.609758, 80.617778), (12.940685, 80.177022), (15.462648, 73.804488), (22.54111, 88.33778), (28.629811, 77.434197), (24.69028, 78.41889), (26.675324, 88.36301), (28.602325, 77.401994), (12.9732, 77.5999), (21.17, 72.83), (30.673383, 76.733924), (18.452663, 73.93104), (12.92639, 80.165), (28.5035, 77.426447), (22.54111, 88.33778), (20.257483, 85.780142), (20.084522, 72.900306)]

# Initialize an empty list to store postal codes
postal_codes = []

# Perform reverse geocoding for each set of coordinates
for lat, lon in coordinates:
    location = geolocator.reverse((lat, lon), exactly_one=True)
    if location is not None:
        postal_code = location.raw.get("address", {}).get("postcode")
        postal_codes.append(postal_code)
    else:
        postal_codes.append(None)  # Add None if no location data is found

# Print the postal codes for each flat
for i, postal_code in enumerate(postal_codes):
    print(f"Flat {i+1} Postal Code:", postal_code)

from geopy.geocoders import Nominatim
import pandas as pd
from tqdm import tqdm
import time
import csv

# Create a geocoder instance
geolocator = Nominatim(user_agent="my_geocoder")
longitude_df = pd.read_csv('/content/Longitude.csv')
latitude_df = pd.read_csv('/content/latitude.csv')

combined_df = pd.concat([longitude_df, latitude_df], axis=1, ignore_index=True)
# print(combined_df)
combined_df.columns=['Latitude','Longitude']
# print(combined_df.head())
coordinates = [(lat, lon) for lat, lon in zip(combined_df['Latitude'], combined_df['Longitude'])]

# Define the longitude and latitude coordinates for the flats
# coordinates = pd.concat([longitude_df, latitude_df], axis=1).to_numpy()
# Initialize an empty list to store postal codes
postal_codes = dict()



# # Perform reverse geocoding for each set of coordinates
for index, (lat,lon) in tqdm(enumerate(coordinates)):
  location = geolocator.reverse((lat, lon), exactly_one=True)
  if location is not None:
    postal_code = location.raw.get("address", {}).get("postcode")
    postal_codes[index] = postal_code
    ###########

    # # Data to append to the CSV file
    # new_data = postal_codes[index]

    # # CSV file path (change this to your file path)
    # csv_file_path = "postals.csv"

    # # Open the CSV file in append mode
    # with open(csv_file_path, mode='a', newline='') as file:
    #     # Create a CSV writer object
    #     writer = csv.writer(file)

    #     # Write the data to the CSV file
    #     writer.writerow(new_data)

    # print("Data appended to the CSV file successfully.")
  else:
    postal_codes[index] = '0'  # Add None if no location data is found
        # Data to append to the CSV file
    # new_data = postal_codes[index]

    # # CSV file path (change this to your file path)
    # csv_file_path = "postals.csv"

    # # Open the CSV file in append mode
    # with open(csv_file_path, mode='a', newline='') as file:
    #     # Create a CSV writer object
    #     writer = csv.writer(file)

    #     # Write the data to the CSV file
    #     writer.writerow(new_data)

    # print("None appended to the CSV file successfully.")

# # Print the postal codes for each flat
# for i, postal_code in enumerate(postal_codes):
#     print(f"Flat {i+1} Postal Code:", postal_code)
# Specify the CSV file name
csv_file = 'postals.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    # Create a CSV writer object
    csv_writer = csv.writer(file)

    # Write the header (keys of the dictionary) as the first row
    header = postal_codes.keys()
    csv_writer.writerow(header)

    # Write the data from the dictionary to the CSV file
    for row in zip(*postal_codes.values()):
        csv_writer.writerow(row)

default_string = "Not Found"

# Iterate through the dictionary and replace None with the default string
for key, value in postal_codes.items():
    if value is None:
        postal_codes[key] = default_string

# Print the modified dictionary
print(postal_codes)

csv_file = 'postals.csv'

# Extract keys and values from the dictionary
keys = list(postal_codes.keys())
values = list(postal_codes.values())

# Create a list of tuples where each tuple contains (key, value)
rows = list(zip(keys, values))

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    # Create a CSV writer object
    csv_writer = csv.writer(file)

    # Write the header row
    csv_writer.writerow(['col1', 'col2'])

    # Write the data (key, value) pairs
    csv_writer.writerows(rows)

print(f'{csv_file} has been created successfully.')

import json

# Convert dictionary to JSON
json_data = json.dumps(postal_codes)

# Print the JSON data
print(json_data)

data = json.loads(json_data)

# Define the default string
default_string = "Null"

# Function to recursively replace null values with the default string
def replace_null_with_default(obj):
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = replace_null_with_default(obj[i])
        return obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = replace_null_with_default(value)
        return obj
    elif obj is None:
        return default_string
    else:
        return obj

# Replace null values with the default string
data_with_default = replace_null_with_default(data)

# Convert the modified data back to JSON
json_data_with_default = json.dumps(data_with_default, indent=2)

# Print the JSON data with null values replaced
print(json_data_with_default)





combined_df.to_csv('latlong.csv', header= None, index = False)

from google.colab import drive
drive.mount('/content/drive')

import pickle

with open('postal_dict.pkl', 'wb') as f:
  pickle.dump(postal_codes, f)

for k, v in postal_codes.items():
  if v == None:
    # do geocode
    print(coordinates[k])
    location = geolocator.reverse(coordinates[k], exactly_one=True).raw.get("address", {}).get("postcode")
    print(location)

import pandas as pd

# Sample DataFrames (replace with your actual DataFrames)
df1 = pd.read_csv('/content/Longitude.csv')
df2 = pd.read_csv('/content/latitude.csv')


# Concatenate the DataFrames and convert to the desired format
combined_df = pd.concat([df1, df2], axis=1, ignore_index=True)
# print(combined_df)
combined_df.columns=['Latitude','Longitude']
# print(combined_df.head())
coordinates = [(lat, lon) for lat, lon in zip(combined_df['Latitude'], combined_df['Longitude'])]

print(coordinates)