from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, json, os, random, time
from tqdm import tqdm

class theStandard:
    def __init__(self, useJSON = False, onlyGetLinks = False):
        self.result = []
        self.useJSON = useJSON
        self.onlyGetLinks = onlyGetLinks
        print('theStandard cawler init')
        
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
            typeOfNews = {
                'business' : 'money-glitz',
                'property' : 'property',
                'property' : 'overseas-property',
                'education' : 'education',
                'education' : 'overseas-education',
                'travel' : 'travel',
                'culture' : 'art-and-culture',
                'health' : 'health-Beauty',
                'technology' : 'technology'
            }
            self._getLinks(typeOfNews)
            self._outputLinksToFile()
        if not self.onlyGetLinks:
            self._cawlerPerPage()
        print('Cawlering done')
        self.driver.quit()

    def _getLinks(self,typeOfNews):
        for type, value in typeOfNews.items():
            self.driver.get(f'https://www.thestandard.com.hk/section-news-list/feature/{value}/')     
            time.sleep(random.uniform(2.0, 3.0))
            mainbox = self.driver.find_element(By.XPATH,'/html/body/div[2]/div/div[1]/div[1]/div')

            totalNumOfNews = 10000
            lastCollenctedNum = 0
            progress = tqdm(total = totalNumOfNews, desc = 'Collecting', unit = 'item', leave=True)
            while True:
                try:
                    self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                    secondbox = mainbox.find_elements(By.CLASS_NAME,'caption')
                    self.driver.find_element(By.CLASS_NAME,'show-more').click()
                    time.sleep(random.uniform(2.0, 5.0))

                    if len(secondbox) >= totalNumOfNews:
                        progress.update(totalNumOfNews - lastCollenctedNum)
                        progress.close()
                        break
                    elif len(secondbox) == lastCollenctedNum:
                        startTime = time.time()
                        while True:
                            self.driver.find_element(By.CLASS_NAME,'show-more').click()
                            time.sleep(random.uniform(2.0, 5.0))
                            if len(mainbox.find_elements(By.CLASS_NAME,'caption')) > lastCollenctedNum:
                                break
                            elif time.time() - startTime > 60:
                                raise Exception
                    else:
                        progress.update(len(secondbox) - lastCollenctedNum)
                        lastCollenctedNum = len(secondbox)
                except:
                    progress.close()
                    print('Error to get more news')
                    print(len(secondbox), 'news link collected')
                    break

            print(f'{len(secondbox)} {type} news items found')
            for data in tqdm(secondbox, desc='Saving', unit='item'):
                self.result.append(
                {
                        'link': data.find_elements(By.TAG_NAME, 'a')[0].get_attribute('href'),      
                        'title': data.find_elements(By.TAG_NAME, 'h1')[0].text,
                        'type': type
                    }
                )
    
    def _cawlerPerPage(self):
        if self.useJSON:
            with open('./linkData/links-theStandard.json', 'r', encoding='utf-8') as file:
                self.result = json.load(file)
        
        skipTo = 3883
        for index, link in enumerate(tqdm(self.result, desc='Cawlering', unit='page')):
            if index < skipTo:
                continue

            try :
                self.driver.get(link['link'])
                time.sleep(random.uniform(2.0, 5.0))

                mainContent = self.driver.find_element(By.CLASS_NAME, 'ts-section')
                pTabs = mainContent.find_elements(By.TAG_NAME,'P')

                content = ''
                for p in pTabs:
                    content+=p.text + ' '

                pageResult = {
                    'title': link['title'],
                    'content': content,
                    'category': link['type'],
                    'link': link['link']
                }
            except Exception as e:
                with open('./error-theStandard', 'a', encoding='utf-8') as file:
                    file.write('\n Error at index ' + str(index) + ' ' + link['link'])
                    file.write('\n' + str(e) + '\n')
                continue
                            
            file_path = './newsData/theStandard.json'
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = []
            data.append(pageResult)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
        
    def _outputLinksToFile(self):
        with open('./linkData/links-theStandard.json', 'w', encoding = 'utf-8') as file:
            json.dump(self.result, file, indent = 4)
        print(f'{len(self.result)} news link items saved')
