import rocks
import numpy as np

rock_num = 14
'''
try:
    # Replace this function with your actual implementation of getType
    print(rocks.Rock(269).diameter.value)
    print(rocks.Rock(1))
except Exception as e:
    print(f"Error in getType function: {e}")
    print(None)
    
'''

with open('demeotax.tab', 'r') as file:
    data = [line.split() for line in file.readlines()]

nums = [int(row[0]) for row in data]
names = [row[1] for row in data]

albedos = []

for asteroid in nums:
    print(asteroid)
    try:
        albedo = rocks.Rock(asteroid).albedo.value
    except:
        albedo = 'nan'
    if albedo != 'nan':
        albedos.append(albedo)
    else:
        albedos.append(-1)

with open('demeo_albedos.txt', 'w') as file:
    for val1, val2 in zip(nums, albedos):
        file.write(f"{val1}\t{val2}\n")

id_ = 1
try:
    rock = rocks.Rock(id_)
    print(rock.albedo.value)
except:
    print('Rock NOT Found')


