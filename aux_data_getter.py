import rocks
import os
'''
nums = []
abs_mags = []
diams = []
albedos = []
for tax in os.listdir('data/cleaned_0.4'):
    for img in os.listdir(f'data/cleaned_0.4/{tax}'):
        number = img[:-4]
        abs_mag = rocks.Rock(number).absolute_magnitude.value
        diameter = rocks.Rock(number).diameter.value
        albedo = rocks.Rock(number).albedo.value

        nums += [number]
        abs_mags += [abs_mag]
        diams += [diameter]
        albedos += [albedo]

print(nums)
print(abs_mags)
print(diams)
print(albedos)

# Open a file to write the data
with open('data/demeo_aux.txt', 'w') as f:
    f.write('num mag diam albedo\n')
    for item1, item2, item3, item4 in zip(nums, abs_mags, diams, albedos):
        f.write(f"{item1} {item2} {item3} {item4}\n")
        
nums = []
abs_mags = []
diams = []
albedos = []
for img in os.listdir(f'data/mithneos_graphs'):
    number = img[:-4]
    abs_mag = rocks.Rock(number).absolute_magnitude.value
    diameter = rocks.Rock(number).diameter.value
    albedo = rocks.Rock(number).albedo.value

    nums += [number]
    abs_mags += [abs_mag]
    diams += [diameter]
    albedos += [albedo]

print(nums)
print(abs_mags)
print(diams)
print(albedos)

# Open a file to write the data
with open('data/mithneos_aux.txt', 'w') as f:
    f.write('num mag diam albedo\n')
    for item1, item2, item3, item4 in zip(nums, abs_mags, diams, albedos):
        f.write(f"{item1} {item2} {item3} {item4}\n")

'''


with open('data/mithneos_aux.txt', 'r') as f:
    lines = f.readlines()
with open('data/mithneos_aux.txt', 'w') as f:
    for line in lines:
        f.write(line.replace('nan', '0'))
