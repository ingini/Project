## parser.py
import datetime
from selenium import webdriver
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np

class AI_assignments():
    def __init__(self):
        super.__init__()
        self.path = os.path.dirname(os.path.realpath(__file__))

        self.number = number

    def time_calc(self,sec):
        if sec == 1:
            #tz = datetime.datetime.now().strftime('%Y%m%d%H%M')
            tz = datetime.datetime.now()
        elif sec == 2:
            #tz = datetime.datetime.now().strftime('%Y%m%d%H%M')
            tz = datetime.datetime.now()
        else:
            tz = '총 걸린 시간을 모르겠네요. 에러났네요~'

        return tz

    def site_parser(self,driver,soup):
        # site url, title, body
        content = []
        df = pd.DataFrame()
        # 각 site 별 xpath
        if self.number == 0:
            # 잡코리아 회사별 합격자들 평균 스펙
            try:
                for order in range(1,10):
                    link_url = driver.find_element_by_xpath(f'//*[@id="container"]/div[2]/div[2]/div[4]/ul/li[{order}]/div[2]/dl/dt/a').get_property('href')
                    link_name = driver.find_element_by_xpath(f'//*[@id="container"]/div[2]/div[2]/div[4]/ul/li[{order}]/div[2]/dl/dt/a').text
                    link_industry = driver.find_element_by_xpath(f'/html/body/div[5]/div[2]/div[2]/div[2]/div[4]/ul/li[{order}]/div[2]/dl/dd[1]/span').text
                    link_sales = driver.find_element_by_xpath(f'/html/body/div[5]/div[2]/div[2]/div[2]/div[4]/ul/li[{order}]/div[2]/dl/dd[2]/span').text
                    link_people = driver.find_element_by_xpath(f'/html/body/div[5]/div[2]/div[2]/div[2]/div[4]/ul/li[{order}]/div[2]/dl/dd[3]/span').text
                    link_point = driver.find_element_by_xpath(f'/html/body/div[5]/div[2]/div[2]/div[2]/div[4]/ul/li[{order}]/div[4]/a/span').text
                    link = [link_name,link_industry,link_sales,link_people,link_point,link_url]
                    content.append(link)
            except:
                print('합격스펙기업 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['기업이름', '산업종류', '매출액','사원수','스펙지수','URL'])
            except:
                print('error : homepage elements path None')

        elif self.number == 1:
            # 뉴욕타임스
            try:
                for order in range(1,10):
                    link_time = driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/main/section/div[1]/div/section/div[1]/ol/li[{order}]/div/div[2]/span').text
                    link_name = driver.find_element_by_xpath(f'//*[@id="stream-panel"]/div[1]/ol/li[{order}]/div/div[1]/a/h2').text
                    link_body = driver.find_element_by_xpath(f'//*[@id="stream-panel"]/div[1]/ol/li[{order}]/div/div[1]/a/p').text
                    link = [link_name,link_body,link_time]
                    content.append(link)
            except:
                print('합격스펙기업 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['Title', 'Body', 'DATE'])
            except:
                print('error : homepage elements path None')

        elif self.number == 2:
            # 학교 장학 페이지 정보
            try:
                for order in range(1,10):
                    link_url = driver.find_element_by_xpath(f'//*[@id="jwxe_main_content"]/div/div/div[1]/div[1]/ul/li[{order}]/dl/dt/a').get_property('href')
                    link_name = driver.find_element_by_xpath(f'//*[@id="jwxe_main_content"]/div/div/div[1]/div[1]/ul/li[{order}]/dl/dt/a').text
                    link_no = driver.find_element_by_xpath(f'/html/body/div[3]/div[2]/div[2]/div/div/div[1]/div[1]/ul/li[{order}]/dl/dd/ul/li[1]').text
                    link_team = driver.find_element_by_xpath(f'/html/body/div[3]/div[2]/div[2]/div/div/div[1]/div[1]/ul/li[{order}]/dl/dd/ul/li[2]').text
                    link_date = driver.find_element_by_xpath(
                        f'/html/body/div[3]/div[2]/div[2]/div/div/div[1]/div[1]/ul/li[{order}]/dl/dd/ul/li[3]').text
                    link_reader = driver.find_element_by_xpath(
                        f'/html/body/div[3]/div[2]/div[2]/div/div/div[1]/div[1]/ul/li[{order}]/dl/dd/ul/li[4]/span').text

                    link = [link_no,link_name,link_team,link_date,link_reader,link_url]
                    content.append(link)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['index','Title', 'Team_name','DATE','조회수','URL'])
            except:
                print('error : homepage elements path None')

        elif self.number == 3:
            # Top list in your area [wines]
            try:
                for order in range(1,10):
                    link_shortnm = driver.find_element_by_xpath(f'//*[@id="toplistsBand"]/div/div/div[2]/div/div/div/div[{order}]/div/a/div[2]/div[1]/div[1]').text
                    link_name = driver.find_element_by_xpath(f'//*[@id="toplistsBand"]/div/div/div[2]/div/div/div/div[{order}]/div/a/div[2]/div[1]/div[2]').text
                    link_origin = driver.find_element_by_xpath(f'//*[@id="toplistsBand"]/div/div/div[2]/div/div/div/div[{order}]/div/a/div[2]/div[2]/div').text
                    link_point = driver.find_element_by_xpath(f'//*[@id="toplistsBand"]/div/div/div[2]/div/div/div/div[{order}]/div/a/div[1]/div[2]/div[1]/div[1]').text
                    link = [link_shortnm,link_name,link_origin,link_point]
                    content.append(link)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['short_name','full_name','원산지','점수'])
            except:
                print('error : homepage elements path None')

        elif self.number == 4:
            # 트게더 우울증 커뮤니티
            # site 의 row- number 를 받아서 crawling
            try:
                range_list = driver.find_elements_by_class_name('article-list-row     ')
                for idx, v in enumerate(range_list):
                    a = str(v.text)
                    a = a.split('\n')
                    print(a)
                    if len(a) ==4:
                        content.append(a)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['index', 'Title', 'nickname','DATE'])
            except:
                print('error : homepage elements path None')

        elif self.number == 5:
            # k-리그 포털 경기 별 매치 요약 ###########################################error
            # 해당 홈페이지는 Ordinal0 << 와 같은 error 발생
            # 구글에 해당 error 를 찾아보니 chrome version error 보임
            # 일단 확실하게 chrome driver version 을 local 과 driver version 동일하게 맞춰서 진행
            # version 낮춰서 test 해보았는데도 Ordinal0, GetHandleVerifier RtlGetAppContainerNamedObjectPath error 뜸. 아예 version 을 확 낮춰야되는것 같음...
            # 96, 98, 100, 101 version test 결과 --> 같은 에러 발생.
            # Ordinal0 error, GetHandleVerifier error, BaseThreadInitThunk error, RtlGetAppContainerNamedObjectPath error 가 떠서
            # homepage 의 url link 와 body 만 받아서 return 하겠음.

            try:
                link_week = driver.find_element_by_xpath('/html/frameset/frame[2]').get_property('src')
                link_body = driver.find_element_by_xpath('/html/body').text
                link = [link_week, link_body]
                content.append(link)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['URL','BODY'])
            except:
                print('error : homepage elements path None')

        elif self.number == 6:
            # 모두닥 병원 별 리뷰
            try:
                for order in range(1,10):
                    link_name = driver.find_element_by_xpath(f'/html/body/div[4]/div/div/div[1]/div[3]/div[{order}]/a/div/div[2]/div[2]/div[1]').text
                    link = link_name.split('\n')
                    content.append(link)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['Name', 'Score', 'Review','Station'])
            except:
                print('error : homepage elements path None')

        elif self.number == 7:
            # 마켓컬리 베스트 상품
            try:
                for order in range(1,10):
                    link_name = driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/div[2]/div/div[2]/div[2]/div[2]/div/ul/li[{order}]/div/a/span[1]').text
                    link_price = driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/div[2]/div/div[2]/div[2]/div[2]/div/ul/li[{order}]/div/a/span[2]').text
                    # 아래는 할인된 상품의 가격을 제외하고 실제 판매하는 가격만 price 로 변경하기 위함
                    try:
                        link_halin = link_price.split('\n')
                        link_price = link_halin[-1]
                    except:
                        continue
                    link_note = driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/div[2]/div/div[2]/div[2]/div[2]/div/ul/li[{order}]/div/a/span[3]').text

                    link = [link_name,link_price,link_note]
                    content.append(link)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['상품이름','가격','설명'])
            except:
                print('error : homepage elements path None')

        elif self.number == 8:
            # Snopes Fact check 포스트
            try:
                for order in range(1,10):
                    link_name = driver.find_element_by_xpath(f'/html/body/div[4]/div/div[1]/main/div[2]/div/article[{order}]/div/span[1]').text
                    link_body = driver.find_element_by_xpath(f'/html/body/div[4]/div/div[1]/main/div[2]/div/article[{order}]/div/span[2]').text
                    link_ox = driver.find_element_by_xpath(f'/html/body/div[4]/div/div[1]/main/div[2]/div/article[{order}]/div/ul/div/div/div/div').text
                    link = [link_name,link_body,link_ox]
                    content.append(link)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['title','body','rep'])
            except:
                print('error : homepage elements path None')

        elif self.number == 9:
            # Buzzfeed 사이트 뉴스기사
            # 가장 hot 한 feed 3 개만 추출
            try:
                for order in range(1,3):
                    link_title = driver.find_element_by_xpath(f'/html/body/div/main/div/div[1]/div[1]/div[1]/section/ul/li[{order}]/div[2]/div/span[1]/span').text
                    link_name = driver.find_element_by_xpath(f'/html/body/div/main/div/div[1]/div[1]/div[1]/section/ul/li[{order}]/div[2]/a/h2').text
                    try:
                        link_time = driver.find_element_by_xpath(f'/html/body/div/main/div/div[1]/div[1]/div[1]/section/ul/li[{order}]/div[2]/div/span[3]').text
                    except:
                        link_time = 'None time'
                    link = [link_title,link_name,link_time]
                    content.append(link)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['Title','body','ago'])
            except:
                print('error : homepage elements path None')

        elif self.number == 10:
            # 성대 뉴스
            try:
                for order in range(1,10):
                    link_no = driver.find_element_by_xpath(f'/html/body/div[3]/div[2]/div[2]/div/div/div/div[2]/table/tbody/tr[{order}]/td[1]').text
                    link_name = driver.find_element_by_xpath(f'/html/body/div[3]/div[2]/div[2]/div/div/div/div[2]/table/tbody/tr[{order}]/td[2]/a').text
                    link_date = driver.find_element_by_xpath(f'/html/body/div[3]/div[2]/div[2]/div/div/div/div[2]/table/tbody/tr[{order}]/td[4]').text
                    link_url = driver.find_element_by_xpath(f'/html/body/div[3]/div[2]/div[2]/div/div/div/div[2]/table/tbody/tr[1]/td[2]/a').get_property('href')
                    link = [link_no,link_name,link_date,link_url]
                    content.append(link)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['no','title','DATE','URL'])
            except:
                print('error : homepage elements path None')

        elif self.number == 11:
            # 잡플래닛 네이버 기업 리뷰 1~5페이지
            driver.find_element_by_xpath('/html/body/div[1]/div[1]/header/div[2]/div/a[1]').click()
            #jobplanet_id = '1dae@g.skku.edu'
            #jobplanet_pw = '@Eorjs1643'
            # jobplanet id / pw 입력 받아서 login
            jobplanet_id = input('멤버쉽 등록된 id 를 입력해주세요 : ')
            jobplanet_pw = input('pw 를 입력해주세요 : ')
            driver.find_element_by_xpath('/html/body/div[1]/div[4]/div/div/div/form/div/div[2]/div/section[2]/fieldset/label[1]/input').send_keys(jobplanet_id)
            driver.find_element_by_xpath(
                '/html/body/div[1]/div[4]/div/div/div/form/div/div[2]/div/section[2]/fieldset/label[2]/input').send_keys(
                jobplanet_pw)
            driver.find_element_by_xpath('/html/body/div[1]/div[4]/div/div/div/form/div/div[2]/div/section[2]/fieldset/button').click()
            driver.get('https://www.jobplanet.co.kr/companies/42217/reviews/%EB%84%A4%EC%9D%B4%EB%B2%84')
            driver.implicitly_wait(3)  # hompaage 대기시간 5초 동안 wait

            # 리뷰 3개만 받겠습니다.
            try:
                for order in range(1,4):
                    link_info = driver.find_element_by_xpath(
                        f'/html/body/div[1]/div[4]/div/div[1]/div[3]/article[2]/div/div/div/section[{order}]/div/div[1]').text
                    link_title = driver.find_element_by_xpath(
                        f'/html/body/div[1]/div[4]/div/div[1]/div[3]/article[2]/div/div/div/section[{order}]/div/div[2]/div/div[1]/h2').text
                    link_adv = driver.find_element_by_xpath(
                        f'/html/body/div[1]/div[4]/div/div[1]/div[3]/article[2]/div/div/div/section[{order}]/div/div[2]/div/dl/dd[1]/span').text
                    link_disadv = driver.find_element_by_xpath(
                        f'/html/body/div[1]/div[4]/div/div[1]/div[3]/article[2]/div/div/div/section[{order}]/div/div[2]/div/dl/dd[2]/span').text
                    link_plz = driver.find_element_by_xpath(
                        f'/html/body/div[1]/div[4]/div/div[1]/div[3]/article[2]/div/div/div/section[{order}]/div/div[2]/div/dl/dd[3]/span').text
                    link = [link_info,link_title,link_adv,link_disadv,link_plz]
                    content.append(link)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['info','title','장점','단점','경영진에 바라는 점'])
            except:
                print('error : homepage elements path None')

        elif self.number == 12:
            # 잡플래닛 채용공고 List
            try:
                for order in range(1,10):
                    link_url = driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/form/div[2]/div[1]/div[2]/div[1]/div[2]/div[2]/div[{order}]/div/div/div[1]/a').get_property('href')
                    link_name = driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/form/div[2]/div[1]/div[2]/div[1]/div[2]/div[2]/div[{order}]/div/div/div[1]/a').text
                    link_corp = driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/form/div[2]/div[1]/div[2]/div[1]/div[2]/div[2]/div[{order}]/div/div/div[2]/div[1]/p/button').text
                    link_note = driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/form/div[2]/div[1]/div[2]/div[1]/div[2]/div[2]/div[{order}]/div/div/div[3]/div[1]').text
                    link_point = driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/form/div[2]/div[1]/div[2]/div[1]/div[2]/div[2]/div[{order}]/div/div/div[2]/div[2]/a[1]/em').text
                    link = [link_name,link_corp,link_note,link_point,link_url]
                    content.append(link)
            except:
                print('페이지 정보 리스트가 없습니다.')
            try:
                df = pd.DataFrame(content, columns=['채용직무','회사이름','회사소개','평점','URL'])
            except:
                print('error : homepage elements path None')

        else:
            print('0 ~ 12 사이의 숫자를 입력해주세요')


        return df

    def crawling_bs4(self,number):
        # 0~12 까지의 싸이트 주소
        self.site_list = ['http://www.jobkorea.co.kr/starter/spec/','https://www.nytimes.com/section/universal/ko',
                     'https://www.skku.edu/skku/campus/skk_comm/notice06.do',
                     'https://www.vivino.com/','https://tgd.kr/c/depressive','https://data.kleague.com/',
                     'https://www.modoodoc.com/q/?search_query=%EC%B9%98%EA%B3%BC',
                     'https://www.kurly.com/shop/goods/goods_list.php?category=029',
                     'https://www.snopes.com/fact-check/', 'https://www.buzzfeed.com/',
                     'https://www.skku.edu/skku/campus/skk_comm/news.do',
                     'https://www.jobplanet.co.kr/companies/42217/reviews/%EB%84%A4%EC%9D%B4%EB%B2%84',
                     'https://www.jobplanet.co.kr/job_postings/search?_rs_act=browse&_rs_con=job&_rs_element=job_postings&occupation_level2_ids%5B%5D=11613']

        # WEB DRIVER CONNECT error 시 chrome driver 경로 설정 필요
        driver = webdriver.Chrome(self.path + '/2021720372/chromedriver_win32/chromedriver.exe')
        #driver = webdriver.Chrome(path + '/2021720372/chromedriver_win32/chromedriver.exe')
        #driver = webdriver.Chrome(path + '/2021720372/chrome_version_98/chromedriver_win32/chromedriver.exe')
        # crawling
        driver.get(self.site_list[number])

        driver.implicitly_wait(3) # hompaage 대기시간 5초 동안 wait
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')

        df = site_parser(driver, soup, number)

        try:
            driver.quit()
        except:
            pass

        now = datetime.datetime.now().strftime('%Y%m%d%H%M')

        df.to_csv(str(now)+'.csv', encoding='utf-8-sig',index=False)

    def Choose_type(self):

        typez = input('계속하시겠습니까? (yes/no) : ')

        return typez

    def crawlings(self):
        start_time = time_calc(1)
        number = int(input('원하시는 site number 를 입력해주세요 (0~12 사이 숫자 입력) : '))
        crawling_bs4(number)
        end_time = time_calc(2)
        final_time = end_time - start_time
        print('웹사이트 주소 : ', str(self.site_list[number]))
        print('총 소요 시간 : ', str(final_time), '(초) 입니다')



    def main(self):
        # 한번만 실행하게 짰다가 pdf file 마지막 실행예를 보고 예시에 맞게 반복실행하게 수정.....
        while True:
            type = Choose_type()

            if type == 'yes':
                crawlings()
            else:
                print('안녕히 가십시오. ')
                break

if __name__ == '__main__':
    master = AI_assignments()
    master.main()