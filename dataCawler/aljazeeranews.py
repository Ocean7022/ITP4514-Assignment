from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, json, os, random
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
            print('Loading web page...')
            self.driver.get(f'https://www.aljazeera.com/{value}')
            print('Web page loaded')
            time.sleep(1)

            totalNumOfNwes = 20000
            lastCollenctedNum = 0
            progress = tqdm(total = totalNumOfNwes, desc = 'Collecting', unit = 'item', leave=True)
            while True:
                news_items = self.driver.find_elements(By.CSS_SELECTOR, 'article.gc')
                if len(news_items) >= totalNumOfNwes:
                    progress.update(totalNumOfNwes - lastCollenctedNum)
                    break
                else:
                    progress.update(len(news_items) - lastCollenctedNum)
                    lastCollenctedNum = len(news_items)

                try:
                    show_more_button = self.driver.find_element(By.CLASS_NAME, 'show-more-button')
                    clickTargrt = show_more_button.find_elements(By.TAG_NAME, 'span')
                    self.driver.execute_script("arguments[0].scrollIntoView();", show_more_button)
                    clickTargrt[1].click()
                    time.sleep(random.uniform(2.0, 5.0))
                except:
                    print(f'\nNo more {type} news')
                    break
            progress.close()

            for item in tqdm(news_items, desc='Saving', unit='item'):
                link = item.find_element(By.CSS_SELECTOR, 'a.u-clickable-card__link').get_attribute('href')
                title = item.find_element(By.CSS_SELECTOR, 'h3.gc__title').text
                self.result.append({'link': link, 'title': title, 'type': type})

    def _cawlerPerPage(self):
        if self.useJSON:
            with open('./linkData/links-aljazeeranews.json', 'r', encoding='utf-8') as file:
                self.result = json.load(file)


        
        pass

    def _outputLinksToFile(self):
        with open('links-aljazeeray.json', 'w', encoding='utf-8') as file:
            json.dump(self.result, file, indent=4)
        print(f'\n{len(self.result)} news link items saved')
