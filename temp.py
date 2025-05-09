from selenium import webdriver
from selenium.webdriver.common.by import By
import time


# Open the URL
url = "https://shop.kimelo.com/sku/cheese-mozzarella-wmlm-feather-shred-nb-45-lb-124254/124254"
response = requests.get(url)

# Optional: Wait for the page to fully load
time.sleep(3)

# Find the image element — update this selector as needed
# The product image seems to be inside <img class="MuiCardMedia-root ...">
image_element = driver.find_element(By.CSS_SELECTOR, '.object-contain.transition-opacity.opacity-0.opacity-100')

# Get the 'src' attribute
image_url = image_element.get_attribute('src')

print("Product Image URL:", image_url)

# Close the browser window
driver.quit()