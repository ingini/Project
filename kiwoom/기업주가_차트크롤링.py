import time
import requests
import subprocess
from selenium import webdriver
from bs4 import BeautifulSoup

news_land = {'title' : [], 'body' : []}
driver = webdriver.Chrome()
code = ['039490']

driver.get("https://finance.naver.com/item/main.nhn?code=" + code[0])
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

title = soup.find('div', class_='rate_info')
title_text = title.get_text(strip=True)
print(title_text)

body = soup.find('div', class_='chart')
print(body)
body_text = body.get('onerror')
print(body_text)
'''with open(basename(body_text), "wb") as f:
    f.write(requests.get(body_text).content)'''



news_land['body'].append(body_text)
news_land['title'].append(title_text)

print(news_land)

'''<img id="img_chart_area" src="https://ssl.pstatic.net/imgfinance/chart/item/area/day/039490.png?sidcode=1573441364833" width="700" height="289" alt="이미지 차트" onerror="this.src='https://ssl.pstatic.net/imgstock/chart3/world2008/error_700x289.png'">'''