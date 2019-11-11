# coding=utf-8                           #
# this is for coding in utf-8            #
####Caution###############################
# This code doesn't work in Web Compiler #
##########################################
#STEP 1
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import webbrowser

#STEP 2
auth_key="Type your own key" #authority key
company_code="024110" #company code
start_date="1990101"

#STEP 3
url = "http://dart.fss.or.kr/api/search.xml?auth="+auth_key+"&crp_cd="+company_code+"&start_dt="+start_date+"&bsn_tp=A001&bsn_tp=A002&bsn_tp=A003"

#STEP 4
resultXML=urlopen(url)  #this is for response of XML
result=resultXML.read() #Using read method

#STEP 5
xmlsoup=BeautifulSoup(result,'html.parser')

#STEP 6
data = pd.DataFrame()

te=xmlsoup.findAll("list")

for t in te:
    temp=pd.DataFrame(([[t.crp_cls.string,t.crp_nm.string,t.crp_cd.string,t.rpt_nm.string,
        t.rcp_no.string,t.flr_nm.string,t.rcp_dt.string, t.rmk.string]]),
        columns=["crp_cls","crp_nm","crp_cd","rpt_nm","rcp_no","flr_nm","rcp_dt","rmk"])
    data=pd.concat([data,temp])

#STEP 7
data=data.reset_index(drop=True)

#OPTIONAL
print(data)
user_num=int(input("몇 번째 보고서를 확인하시겠습니까?"))
url_user="http://dart.fss.or.kr/dsaf001/main.do?rcpNo="+data['rcp_no'][user_num]
webbrowser.open(url_user)

# 재무제표 크롤링
url2="http://dart.fss.or.kr/report/viewer.do?rcpNo=20170811001153&dcmNo=5746981&eleId=15&offset=297450&length=378975&dtd=dart3.xsd"

report=urlopen(url2)
r=report.read()
xmlsoup=BeautifulSoup(r,'html.parser')
body=xmlsoup.find("body")
table=body.find_all("table")
p = parser.make2d(table[3])

sheet = pd.DataFrame(p[2:], columns=["구분","38기반기_3개월","38기반기_누적","37기반기_3개월","37기반기_누적"])