import scrapy

class UsnewsSpider(scrapy.Spider):
    name = 'usnews'
    allowed_domains = ['money.usnews.com']
    start_urls = ['https://money.usnews.com/investing/news/articles/2023-11-16/tiktok-joins-meta-in-appealing-against-eu-gatekeeper-status']

    def parse(self, response):
        title = response.xpath('//h1/text()').get()
        content = ''.join(response.xpath('//article//p/text()').getall())
        yield {
            'title': title,
            'content': content
        }
