import time
from selenium import webdriver
from bs4 import BeautifulSoup

news_land = {'title': [], 'body': []}
driver = webdriver.Chrome()
code = ['024110']

driver.get(f'https://search.naver.com/search.naver?query={code[0]}&where=news&ie=utf8&sm=nws_hty')

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

title = soup.find('ul', class_='type01')
title_text = title.get_text(strip=True)

body = soup.find('div', class_='sub_pack')
body_text = body.get_text(strip=True)

news_land['body'].append(body_text)
news_land['title'].append(title_text)

print(news_land)

