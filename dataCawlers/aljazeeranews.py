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
        self.articles = []

    def start(self):
        chrome_options = Options()
        chrome_options.add_experimental_option('detach', True)
        chrome_options.add_argument("--log-level=3")
        #chrome_options.add_argument("--headless")
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
                    progress.close()
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
                    # retry 10 times
                    for i in range(0, 10):
                        try:
                            show_more_button = self.driver.find_element(By.CLASS_NAME, 'show-more-button')
                            clickTargrt = show_more_button.find_elements(By.TAG_NAME, 'span')
                            self.driver.execute_script("arguments[0].scrollIntoView();", show_more_button)
                            clickTargrt[1].click()
                            time.sleep(random.uniform(2.0, 5.0))
                        except:
                            pass
                    progress.close()
                    print(f'Error to load more news, {len(news_items)} news collected')
                    break

            for item in tqdm(news_items, desc='Saving', unit='item'):
                link = item.find_element(By.CSS_SELECTOR, 'a.u-clickable-card__link').get_attribute('href')
                title = item.find_element(By.CSS_SELECTOR, 'h3.gc__title').text
                self.result.append({'link': link, 'title': title, 'type': type})

    def _cawlerPerPage(self):
        if self.useJSON:
            with open('./linkData/links-aljazeeray.json', 'r', encoding='utf-8') as file:
                self.result = json.load(file)

        for item in tqdm(self.result, desc='Cawlering', unit='page'):
            self._getArticleContent(item)

    def _getArticleContent(self, link_data):
        try:
            self.driver.get(link_data['link'])
            time.sleep(random.uniform(2.0, 5.0))

            content_area = self.driver.find_element(By.CLASS_NAME, 'wysiwyg--all-content')
            paragraphs = content_area.find_elements(By.TAG_NAME, 'p')
            content = ' '.join([p.text for p in paragraphs if p.text])

            pageResult = {
                "title": link_data['title'],
                "content": content,
                "category": link_data['type'],
                'link': link_data['link'],
            }

            file_path = './newsData/aljazeerayData.json'
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = []
            data.append(pageResult)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            with open('./error-theStandard.txt', 'a', encoding='utf-8') as file:
                file.write('\n Error at ' + link_data['link'])
                file.write('\n' + str(e) + '\n')

    def _outputLinksToFile(self):
        with open('./linkData/links-aljazeeray.json', 'w', encoding='utf-8') as file:
            json.dump(self.result, file, indent=4)
        print(f'\n{len(self.result)} news link items saved')
