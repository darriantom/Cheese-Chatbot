import time
import os
import json
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchWindowException, StaleElementReferenceException
from concurrent.futures import ThreadPoolExecutor
import threading

cheeses = []
image_lock = threading.Lock()

def download_image(image_url, product_name):
    if not image_url or not image_url.startswith("http"):
        return "N/A"
        
    try:
        # Create a safe filename from the product name
        safe_name = "".join(c for c in product_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}.jpg"
        output_path = os.path.join("downloaded_images", filename)
        
        # Check if image already exists
        if os.path.exists(output_path):
            return output_path
            
        # Download and save image
        response = requests.get(image_url, timeout=10)
        with image_lock:
            with open(output_path, "wb") as f:
                f.write(response.content)
        return output_path
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return "N/A"

def scrape_links(url, index):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Initialize variables
    size = "N/A"
    weight = "N/A"
    image_url = "N/A"
    number1 = "N/A"
    try:
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 3)
        
        print(f"Navigating to: {url}")
        driver.get(url)
        
        # Wait for main elements to load
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".chakra-heading.css-18j379d")))
        
        # Get all required data in one go
        product_name = driver.find_element(By.CSS_SELECTOR, ".chakra-heading.css-18j379d").text
        company_name = driver.find_element(By.CSS_SELECTOR, ".chakra-text.css-drbcjm").text
        
        # Get numbers
        nums = driver.find_elements(By.CSS_SELECTOR, ".chakra-text.css-0")
        num1 = nums[0].text if len(nums) > 0 else "N/A"
        num2 = nums[1].text if len(nums) > 1 else "N/A"
        
        # Get price and unit price
        unit_price = driver.find_element(By.CSS_SELECTOR, ".chakra-badge.css-1mwp5d1").text
        
        # Get other info
        otherinfo = driver.find_elements(By.CSS_SELECTOR, ".css-1eyncsv")
        if len(otherinfo) == 3:
            size = otherinfo[1].text
            weight = otherinfo[2].text
        else:
            size = otherinfo[3].text if len(otherinfo) > 3 else "N/A"
            weight = otherinfo[5].text if len(otherinfo) > 5 else "N/A"
        
        # Get image URL
        image_elements = driver.find_elements(By.CSS_SELECTOR, '.object-contain.transition-opacity.opacity-0.opacity-100')
        if image_elements:
            image_url = image_elements[0].get_attribute("src")
        
        numbers = driver.find_elements(By.CSS_SELECTOR, ".chakra-text.css-0")
        for index , number in enumerate(numbers):
            if index == 6:
                number1 = number
                print(f"number1: {number1.text}")
        # Create cheese entry
        cheese_entry = {
            "product_name": product_name,
            "company_name": company_name,
            "price": float(num2.replace("$","")),
            "Unit": num1,
            "Cost per pound": float(unit_price.split('/')[0].replace("$","")),
            "standard": size,
            "weight(pound)": float(weight.split(' ')[0]),
            "SKU": int(number1.text),
            "UPC": int(number1.text),
            "image_url": image_url
        }
        print(f"cheese_entry: {cheese_entry}")
        # Download image in a separate thread
        # if image_url != "N/A":
        #     with ThreadPoolExecutor(max_workers=1) as executor:
        #         future = executor.submit(download_image, image_url, product_name)
        #         cheese_entry["image_path"] = future.result()
        
        cheeses.append(cheese_entry)
        print(f"Processed: {product_name}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        try:
            driver.quit()
        except:
            pass

def scrape_cheese():
    # Create images directory once
    os.makedirs("downloaded_images", exist_ok=True)
    
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless') 
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 5)
        
        urls = ["https://shop.kimelo.com/department/cheese/3365?page=" + str(i) for i in range(1, 6)]
        for url in urls:
            print(f"Navigating to: {url}")
            driver.get(url)
            
            # Wait for the page to load and products to be visible
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".relative.css-1bpq4gx")))
            
            # Give extra time for dynamic content
            time.sleep(2)
            
            # Find all product cards on the page
            links = driver.find_elements(By.CSS_SELECTOR, '.chakra-card.group.css-5pmr4x')
            
            # Process links in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.map(lambda x: scrape_links(x.get_attribute('href'), 0), links)
            
        print("\nSaving data to cheese_data.json...")
        with open("cheese_data.json", "w", encoding="utf-8") as f:
            json.dump(cheeses, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully scraped {len(cheeses)} cheese products")
        
    except TimeoutException:
        print("Timeout waiting for elements to load")
    except NoSuchWindowException:
        print("Browser window was closed unexpectedly")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    scrape_cheese()