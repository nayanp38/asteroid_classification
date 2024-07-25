from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException


with open('demeotax.tab', 'r') as file:
    data = [line.split() for line in file.readlines()]

nums = [row[0] for row in data]
names = [row[1] for row in data]

full_ids = []

for i in range(len(nums)):
    if names[i] != '-':
        full_ids.append(nums[i] + ' ' + names[i])
    else:
        full_ids.append(nums[i])

print(full_ids)
driver = webdriver.Chrome()
albedos = []

for asteroid in full_ids:
    driver.get('https://sbnapps.psi.edu/ferret/SimpleSearch/form.action')
    asteroid_search = driver.find_element(By.ID, 'results_targetName')
    asteroid_search.send_keys(asteroid)
    driver.find_element(By.ID, 'results_0').click()
    driver.implicitly_wait(10)

    tables = driver.find_elements(By.XPATH, "//table")

    albedo = 'nan'

    for table in tables:
        # Check if the desired text is present in the table
        if 'MIMPS ALBEDO' in table.text:
            print('albedo found')

            # Find the first row of the table body
            try:
                cell = table.find_element(By.XPATH, ".//tbody/tr/td")
                # Get the value in the desired column
                albedo = cell.text
            except:
                albedo = 'nan'
    print(asteroid + ": " + albedo)
    albedos.append(albedo)

print(albedos)
with open('scraped_albedos.txt', 'w') as file:
    # Write header (optional)
    # Loop through the arrays and write to the file
    for val1, val2 in zip(nums, albedos):
        file.write(f"{val1}\t{val2}\n")