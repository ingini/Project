import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import  *
from PyQt5.QAxContainer import *

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.kiwoom = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")
        self.kiwoom.dynamicCall("CommConnect()")

        self.setWindowTitle("종목 코드")
        self.setGeometry(300, 300, 300, 150)

        btn1 = QPushButton("종목코드 얻기", self)
        btn1.move(190, 10)
        btn1.clicked.connect(self.btn1_clicked)

        self.listWidget = QListWidget(self)
        self.listWidget.setGeometry(10, 10, 170, 130)

    def btn1_clicked(self):
        ret = self.kiwoom.dynamicCall("GetCodeListByMarket(QString)", ["0"])
        kospi_code_list = ret.split(';')
        kospi_code_name_list = []

        for x in kospi_code_list:
            name = self.kiwoom.dynamicCall("GetMasterCodeName(QString)", [x])
            kospi_code_name_list.append(x + " : " + name)

        self.listWidget.addItems(kospi_code_name_list)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    sys.exit(app.exec_())

ret = self.kiwoom.dynamicCall("GetCodeListByMarket(QString)", ["0"]) #증권시장종목코드

kospi_code_list = ret.split(';') #세미콜론으로 종목구분

kospi_code_name_list = []
for x in kospi_code_list:  #종목명과 종목코드 를 하나의 문자열로 따오는 반복문
    name = self.kiwoom.dynamicCall("GetMasterCodeName(QString)", [x])
    kospi_code_name_list.append(x + " : " + name)


self.listWidget.addItems(kospi_code_name_list)#파이썬 리스트에 있는 것을 위젯에 추가