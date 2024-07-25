import pandas as pd

# Specify the path to your .tab file
tab_file_path = 'demeotax.tab'

# Read the .tab file into a DataFrame
# Make sure to set the delimiter parameter to '\t' for tab-separated files
df = pd.read_csv(tab_file_path, delimiter='\ \ \ ')
pd.set_option('display.max_columns', None)  # Set to None to display all columns
'''
# Display the DataFrame
df['Name'] = df['Name'].astype(str)
# Create a new column 'clean_name' with names containing only letters
df['clean_name'] = df['Name'].str.replace(r'[^a-zA-Z]', '')
'''
# Display the updated DataFrame
print(df['Spec.type'])