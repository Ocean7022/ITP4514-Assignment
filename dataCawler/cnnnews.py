from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, json, os, random
from tqdm import tqdm

class cnnnews:
    def __init__(self, useJSON = False, onlyGetLinks = False):
        self.result = []
        self.useJSON = useJSON
        self.onlyGetLinks = onlyGetLinks
        
    def start(self):
        chrome_options = Options()
        chrome_options.add_experimental_option('detach', True)
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument("--headless")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        if self.useJSON:
            print('Using JSON file')
        if self.onlyGetLinks:
            print('Only get links')

        if not self.useJSON:
            self.getPoliticsLinks()
            #typeOfNews = ['health', 'business', 'politics', 'culture-matters']
            #self._getLinks(typeOfNews)
            #self._outputLinksToFile()
        if not self.onlyGetLinks:
            self._cawlerPerPage()
        print('Cawlering done')
        self.driver.quit()

    def _getLinks(self, typeOfNews):
        for type in typeOfNews:
            self.driver.get(f'https://www.nbcnews.com/{type}')
            time.sleep(random.uniform(1.0, 3.0))
            for times in range(0, 100):
                self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                mainBox = self.driver.find_element(By.CLASS_NAME, 'styles_itemsContainer__saJYW')
                newsBoxs = mainBox.find_elements(By.CLASS_NAME, 'wide-tease-item__wrapper')
                print(f'\r{len(newsBoxs)} {type} news items found', end='')
                try:
                    self.driver.find_element(By.CLASS_NAME, 'styles_loadMoreWrapper__pOldr').find_element(By.TAG_NAME, 'button').click()
                    time.sleep(random.uniform(1.0, 4.0))
                except:
                    print(f'\nNo more {type} news')
                    break

            for data in tqdm(newsBoxs, desc='Saving', unit='item'):
                self.result.append(
                    {
                        'link': data.find_elements(By.TAG_NAME, 'a')[2].get_attribute('href'),      
                        'title': data.find_elements(By.TAG_NAME, 'a')[2].text,
                        'type': type,
                    }
                )
    
    def _cawlerPerPage(self):
        if self.useJSON:
            with open('./linkData/links-cnnnews.json', 'r', encoding='utf-8') as file:
                self.result = json.load(file)

        for link in tqdm(self.result, desc='Cawlering', unit='page'):
            self.driver.get(link['link'])
            time.sleep(1)

            mainContent = self.driver.find_elements(By.CLASS_NAME, 'article-body__content')
            pTab = []
            for content in mainContent:
                pTab.append(content.find_elements(By.TAG_NAME, 'p'))

            content = ''
            for p in pTab:
                for data in p:
                    content += data.text + ' '

            pageResult = {
                'title': link['title'],
                'content': content,
                'category': 'culture' if link['type'] == 'culture-matters' else link['type'],
                'link': link['link']
            }

            file_path = './newsData/cnnnewsData.json'
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = []
            data.append(pageResult)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)

    def _outputLinksToFile(self, linkData):
        file_path = './linkData/links-cnnnews.json'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        else:
            data = []
        data.append(linkData)
        with open(file_path, 'w', encoding = 'utf-8') as file:
            json.dump(data, file, indent = 4)
        print(f'{len(data)} news link items saved')

    def getPoliticsLinks(self):
        size = 100
        totalNumOfNews = 15000
        progress = tqdm(total = totalNumOfNews, desc = 'Collecting', unit = 'item', leave=True)
        for page in range(0, 100):
            try:
                self.driver.get(f'https://edition.cnn.com/search?q=politics&from={page * size - size}&size={size}&page={page}&sort=newest&types=article&section=')
                time.sleep(random.uniform(3.0, 5.0))
                mainBox = self.driver.find_element(By.CLASS_NAME, 'container_list-images-with-description__field-links')
                divs = mainBox.find_elements(By.CLASS_NAME, 'container__item--type-media-image')

                for div in divs:
                    link = div.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    if '/politics/' not in link:
                        continue

                    data = {
                        'link': link,
                        'title': div.find_element(By.TAG_NAME, 'span').text,
                        'type': 'politics'
                    }
                    file_path = './linkData/links-cnnnews.json'
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                    else:
                        data = []
                    data.append(data)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(data, file, indent=4)
                    progress.update(1)
                
                if len(data) > totalNumOfNews:
                    progress.close()
                    break
            except:
                progress.close()
                print('Error to get more news')
                break
        