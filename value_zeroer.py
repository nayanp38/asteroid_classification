input_file_path = '500_albedos_raw'
output_file_path = '500_albedos_zeroed'

# Read content from the input file
with open(input_file_path, 'r') as input_file:
    lines = input_file.readlines()

# Replace "nan" with 0 in each line
lines = [line.replace("nan", "0") for line in lines]

# Write the updated content to the output file
with open(output_file_path, 'w') as output_file:
    output_file.writelines(lines)

print("Replacement complete. Updated content saved to:", output_file_path)
