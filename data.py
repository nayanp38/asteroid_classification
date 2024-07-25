import pandas as pd
from sklearn.model_selection import train_test_split
import re


# Load the CSV file
csv_file_path = 'Asteroid Selection.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path)


def evaluate_expression(expression):
    try:
        # Use regular expression to extract two numbers and the division sign
        matches = re.match(r'([\d,]+(?:\.\d+)?)\s*÷\s*([\d,]+(?:\.\d+)?)', expression)
        if matches:
            num1 = float(matches.group(1).replace(',', '.'))
            num2 = float(matches.group(2).replace(',', '.'))

            # Replace the expression with the result of the division
            return str(num1 / num2)
    except Exception as e:
        print(f"Error while evaluating expression: {expression}, {e}")

    # Return the original expression if it cannot be evaluated
    return expression


# Assuming you have a column named 'Spec.type' for the asteroid type
# and a column named 'Diameter (km)' for the diameter labels

df['Diameter (km)'] = df['Diameter (km)'].map(evaluate_expression)
df['Diameter (km)'] = df['Diameter (km)'].str.replace(',', '.').astype(float)

# Extract features (X) and labels (y)
X = df['Diameter (km)']  # Features
y = df['Spec.type']  # Labels

# Convert asteroid types to numerical values using one-hot encoding
y = pd.get_dummies(y, prefix='Spec.type', drop_first=True)


# Save the processed data if needed
processed_csv_path = 'processed_csv/processed_data.csv'
processed_df = pd.concat([X, y], axis=1)
processed_df.to_csv(processed_csv_path, index=False)