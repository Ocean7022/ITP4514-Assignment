from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, json

class nbcnews:
    def __init__(self):
        self.result = []
        
    def start(self):
        chrome_options = Options()
        chrome_options.add_experimental_option('detach', True)
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        typeOfNews = ['health', 'business', 'politics', 'culture-matters']
        self._getLinks(typeOfNews)
        self._outputLinksToFile()
        self._cawlerPerPage()
        self.driver.quit()

    def _getLinks(self, typeOfNews):
        for type in typeOfNews:
            self.driver.get(f'https://www.nbcnews.com/{type}')
            time.sleep(2)
            for times in range(0, 100):
                self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                mainBox = self.driver.find_element(By.CLASS_NAME, 'styles_itemsContainer__saJYW')
                newsBoxs = mainBox.find_elements(By.CLASS_NAME, 'wide-tease-item__wrapper')
                print(f'\r{len(newsBoxs)} {type} news items found', end='')
                try:
                    self.driver.find_element(By.CLASS_NAME, 'styles_loadMoreWrapper__pOldr').find_element(By.TAG_NAME, 'button').click()
                    time.sleep(1)
                except:
                    print(f'\nNo more {type} news')
                    break

            for data in newsBoxs:
                self.result.append(
                    {
                        'link': data.find_elements(By.TAG_NAME, 'a')[2].get_attribute('href'),      
                        'title': data.find_elements(By.TAG_NAME, 'a')[2].text,
                        'type': type,
                    }
                )
    
    def _cawlerPerPage(self):
        pass

    def _outputLinksToFile(self):
        with open('./linkData/links-nbcnews.json', 'w', encoding = 'utf-8') as file:
            json.dump(self.result, file, indent = 4)
        print(f'{len(self.result)} news link items saved')
