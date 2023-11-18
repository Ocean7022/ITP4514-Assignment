from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, json, os
from tqdm import tqdm

class aljazeeranews:
    def __init__(self, useJSON = False, onlyGetLinks = False):
        self.result = []
        self.useJSON = useJSON
        self.onlyGetLinks = onlyGetLinks

    def start(self):
        chrome_options = Options()
        chrome_options.add_experimental_option('detach', True)
        chrome_options.add_argument("--log-level=3")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        if self.useJSON:
            print('Using JSON file')
        if self.onlyGetLinks:
            print('Only get links')

        if not self.useJSON:
            typeOfNews = {
                'sport': 'sport',
                'business': 'economy'
            }
            self._getLinks(typeOfNews)
            self._outputLinksToFile()
        if not self.onlyGetLinks:
            self._cawlerPerPage()
        print('Cawlering done')
        self.driver.quit()

    def _getLinks(self, typeOfNews):
        for type, value in typeOfNews.items():
            self.driver.get(f'https://www.aljazeera.com/{value}')
            time.sleep(1)

            for times in range(0, 100):
                news_items = self.driver.find_elements(By.CSS_SELECTOR, 'article.gc')
                print(f'\r{len(news_items)} {type} news items found', end='')

                try:
                    show_more_button = self.driver.find_element(By.CLASS_NAME, 'show-more-button')
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", show_more_button)
                    self.driver.execute_script("arguments[0].click();", show_more_button)
                    time.sleep(1)
                except Exception as e:
                    print(f'\nNo more {type} news')
                    break

            for item in news_items:
                link = item.find_element(By.CSS_SELECTOR, 'a.u-clickable-card__link').get_attribute('href')
                title = item.find_element(By.CSS_SELECTOR, 'h3.gc__title').text
                self.result.append({'link': link, 'title': title, 'type': type})

    def _cawlerPerPage(self):
        pass

    def _outputLinksToFile(self):
        with open('links-aljazeeray.json', 'w', encoding='utf-8') as file:
            json.dump(self.result, file, indent=4)
        print(f'\n{len(self.result)} news link items saved')
