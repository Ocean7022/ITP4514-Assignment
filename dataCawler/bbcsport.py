from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, json,os

class bbcsport:
    def __init__(self):
        self.result = []
        
    def start(self):
        chrome_options = Options()
        
        chrome_options.add_experimental_option('detach', True)
        chrome_options.add_argument("--log-level=3")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        #,'editorial','china','finance','central-station','world','local','sports']
        typeOfNews = ['top-news']
        self._getLinks(typeOfNews)
        self._outputLinksToFile()
        self._cawlerPerPage()
        self.driver.quit()

    def _getLinks(self,typeOfNews):
        for type in typeOfNews:
            self.driver.get(f'https://www.thestandard.com.hk/section-news-list/section/{type}/')
            time.sleep(2)
            mainbox = self.driver.find_element(By.XPATH,'/html/body/div[2]/div/div[1]/div[1]/div')
            for times in range(1, 2):
                self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                secondbox =mainbox.find_elements(By.CLASS_NAME,'caption')
                print(f'\r{len(secondbox)} {type} news items found', end='')
                try:
                    self.driver.find_element(By.CLASS_NAME,'show-more').click()
                    time.sleep(1)
                except:
                    print(f'\nNo more news')
                    break

              
            
            
            #divs = mainbox.find_elements(By.TAG_NAME, 'div')
            #print(len(divs))
            

            for data in secondbox:
                
                    
                    
                
                self.result.append(
                   {
                        'link': data.find_elements(By.TAG_NAME, 'a')[0].get_attribute('href'),      
                        'title': data.find_elements(By.TAG_NAME, 'h1')[0].text,
                        'type': type
                    }
                )
    
    def _cawlerPerPage(self):
        for link in self.result:
            self.driver.get(link['link'])
            time.sleep(2)

            mainContent = self.driver.find_element(By.CLASS_NAME, 'ts-section')
            pTabs = mainContent.find_elements(By.TAG_NAME,'P')
            
                
            

            content = ''
            for p in pTabs:
                content+=p.text
                #print(p.text)
            #    for data in p:
            #        content += data.text
            #print(content)
                
            

            pageResult = {
                'title': link['title'],
                'content': content,
                'category': link['type']
            }

            file_path = './newsData/ABCD.json'
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = []
            data.append(pageResult)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
        
        

    def _outputLinksToFile(self):
        with open('./linkData/links-ABC.json', 'w', encoding = 'utf-8') as file:
            json.dump(self.result, file, indent = 4)
        print(f'{len(self.result)} news link items saved')
    
        
