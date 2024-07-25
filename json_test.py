import json
import pandas as pd

# Specify the path to your JSON file
json_file_path = 'neap15_extended.json'

# Read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

# Display the DataFrame
pd.set_option('display.max_columns', None)  # Set to None to display all columns
print(df)


