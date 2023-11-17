from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, json

class aljazeeranewsRead:
    def __init__(self):
        self.articles = []

    def start(self):
        # 读取原始 JSON 文件
        with open('links-aljazeeranews.json', 'r', encoding='utf-8') as file:
            links_data = json.load(file)

        chrome_options = Options()
        chrome_options.add_experimental_option('detach', True)
        chrome_options.add_argument("--log-level=3")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 从每个链接获取文章内容
        for item in links_data:
            self._getArticleContent(item)

        self.driver.quit()

        # 将文章数据保存到新的 JSON 文件
        with open('result.json', 'w', encoding='utf-8') as file:
            json.dump(self.articles, file, indent=4)

    def _getArticleContent(self, link_data):
        self.driver.get(link_data['link'])
        time.sleep(2)

        try:
            # 提取文章内容
            content_area = self.driver.find_element(By.CSS_SELECTOR, 'main#main-content-area')
            paragraphs = content_area.find_elements(By.CSS_SELECTOR, 'p')
            content = ' '.join([p.text for p in paragraphs if p.text])

            # 添加到文章列表
            self.articles.append({
                "title": link_data['title'],
                "content": content,
                "category": link_data['type']
            })

        except Exception as e:
            print(f'Error while processing {link_data["link"]}: {e}')

scraper = aljazeeranewsRead()
scraper.start()
