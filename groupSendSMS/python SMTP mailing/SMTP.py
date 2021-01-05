import os
import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
import datetime
from email import encoders

naver_server = smtplib.SMTP_SSL('smtp.gmail.com',465)
naver_server.login('ID','PW')

msg = MIMEBase('multipart','mixed')
todays = datetime.datetime.now()
msg['Subject'] = f'해외 시세 DATA %s' % todays

cont = MIMEText('test 대건 \n 안녕하세요', 'plain','utf-8')

cont['From'] = '보내는 주소'
cont['To'] = '받는 주소'

msg.attach(cont)

path = r'파일경로'
part = MIMEBase("application","octet-stream")
part.set_payload(open(path,'rb').read())

encoders.encode_base64(part)
part.add_header('Content-Disposition','attachment; filename="%s"'% os.path.basename(path))
msg.attach(part)


naver_server.sendmail('보내는 메일 주소','받는 메일 주소', msg.as_string())
naver_server.quit()

####### ERROR 시 체크해야 될 인증 문제 2가지 #############
# 보안 수준이 낮은 앱의 액세스 #### 사용함 << 필수 #####
#https://myaccount.google.com/lesssecureapps?rapt=AEjHL4Nyhpq3EfKNUMKEWdQqOKLWTAnt57ZQG5nni5VTXcc2TnaAVOysD3jgGMNkn6gIMpvgW3OxBUA2C1dOXeh0H7p80Bz9OQ
# linux 환경에서 smtp 사용하려면 보내는 메일의 2단계 인증 필수
# https://greensul.tistory.com/31