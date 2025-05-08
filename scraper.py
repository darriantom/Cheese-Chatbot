import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchWindowException, StaleElementReferenceException

def scrape_cheese():
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless') 
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 10)
        
        url = "https://shop.kimelo.com/department/cheese/3365"
        print(f"Navigating to: {url}")
        driver.get(url)
        
        # Wait for the page to load and products to be visible
        print("Waiting for product cards to load...")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".relative.css-1bpq4gx")))
        
        # Give extra time for dynamic content
        time.sleep(3)
        
        # Find all product cards on the page
        print("Searching for product cards...")
        products = driver.find_elements(By.CSS_SELECTOR, ".relative.css-1bpq4gx")
        print(f"Found {len(products)} product cards")
        
        cheeses = []
        for index, product in enumerate(products, 1):
            try:
                print(f"\nProcessing product {index} of {len(products)}")
                # Find product details within the current product card
                try:
                    description = product.find_element(By.CSS_SELECTOR, ".chakra-text.css-pbtft").text
                    print(f"Found product: {description}")
                except:
                    description = "No description found"
                    print("No description found")
                
                try:
                    brand = product.find_element(By.CSS_SELECTOR, ".chakra-text.css-w6ttxb").text
                    print(f"Brand: {brand}")
                except:
                    brand = "No brand found"
                    print("No brand found")
                
                try:
                    total_price = product.find_element(By.CSS_SELECTOR, ".chakra-text.css-1vhzs63").text
                    print(f"Total Price: {total_price}")
                except:
                    total_price = "No price found"
                    print("No price found")
                
                try:
                    unit_price = product.find_element(By.CSS_SELECTOR, ".chakra-badge.css-ff7g47").text
                    print(f"Unit Price: {unit_price}")
                except:
                    unit_price = "No unit price found"
                    print("No unit price found")
                
                cheeses.append({
                    "description": description,
                    "brand": brand,
                    "total_price": total_price,
                    "unit_price": unit_price
                })
                
            except StaleElementReferenceException:
                print(f"Product {index} became stale, skipping...")
                continue
            except Exception as e:
                print(f"Error processing product {index}: {str(e)}")
                continue
        
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