from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time

# 设置Chrome选项（如果需要）
chrome_options = Options()
# 例如，要在无头模式下运行Chrome，取消注释以下行
# chrome_options.add_argument("--headless")

# 使用WebDriver Manager自动管理驱动程序版本
service = Service(ChromeDriverManager().install())

# 创建WebDriver实例
driver = webdriver.Chrome(service=service, options=chrome_options)

# 打开Google首页
driver.get("http://www.google.com")

time.sleep(5)

# 找到搜索框
#search_box = driver.find_element_by_name("q")  # Google搜索框的name属性是'q'

# 输入搜索内容并提交
#search_box.send_keys("Python")
#search_box.send_keys(Keys.RETURN)  # 模拟按下回车键

# 等待页面加载
driver.implicitly_wait(10)  # 等待10秒

# 关闭浏览器
driver.quit()
