from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
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
        #chrome_options.add_argument("--headless")
        caps = DesiredCapabilities().CHROME
        caps["pageLoadStrategy"] = "none"
        chrome_options.add_experimental_option('prefs', {'profile.managed_default_content_settings.images': 2})
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        chrome_options.capabilities.update(caps)
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        if self.useJSON:
            print('Using JSON file')
        if self.onlyGetLinks:
            print('Only get links')

        if not self.useJSON:
            self._getPoliticsLinks()
        if not self.onlyGetLinks:
            self._cawlerPerPage()
        print('Cawlering done')
        self.driver.quit()

    def _getPoliticsLinks(self):
        print('Loading web page')
        self.driver.get('https://edition.cnn.com/search?q=')
        time.sleep(random.uniform(5.0, 7.0))
        print('Web page loaded')

        size = 100
        totalNumOfNews = 15000
        progress = tqdm(total = totalNumOfNews, desc = 'Collecting', unit = 'item', leave=True)
        for page in range(1, 100):
            try:
                self.driver.get(f'https://edition.cnn.com/search?q=politics&from={page * size - size}&size={size}&page={page}&sort=newest&types=article&section=')
                time.sleep(random.uniform(3.0, 5.0))
                mainBox = self.driver.find_element(By.CLASS_NAME, 'container_list-images-with-description__field-links')
                divs = mainBox.find_elements(By.CLASS_NAME, 'container__item--type-media-image')

                file_path = './linkData/links-cnnnews.json'
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        oldData = json.load(file)
                else:
                    oldData = []

                for div in divs:
                    link = div.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    if '/politics/' not in link:
                        continue

                    data = {
                        'link': link,
                        'title': div.find_element(By.TAG_NAME, 'span').text,
                        'type': 'politics'
                    }
                    oldData.append(data)
                    progress.update(1)

                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(oldData, file, indent=4)
                
                if len(oldData) > totalNumOfNews:
                    progress.close()
                    break
            except Exception as e:
                progress.close()
                print('Error to get more news')
                with open('./error-cnn.txt', 'a', encoding='utf-8') as file:
                    file.write('\n' + str(e) + '\n')
                break

    def _cawlerPerPage(self):
        if self.useJSON:
            with open('./linkData/links-cnnnews.json', 'r', encoding='utf-8') as file:
                self.result = json.load(file)

        for link in tqdm(self.result, desc='Cawlering', unit='page', ncols=100):
            try:
                self.driver.get(link['link'])
                time.sleep(random.uniform(2.0, 4.0))
                self.driver.execute_script("window.stop();")
                content = self.driver.find_element(By.CLASS_NAME, 'article__content').text         

                pageResult = {
                    'title': link['title'],
                    'content': content,
                    'category': 'politics',
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
            except Exception as e:
                with open('./error-cnnnews.txt', 'a', encoding='utf-8') as file:
                    file.write('\n Error at ' + link['link'])
                    file.write('\n' + str(e) + '\n')
                continue

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

       