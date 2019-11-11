키움증권 Open API, 텐서플로(Tensorflow), 케라스(Keras)를 동시에 사용하기 위해서는, 

64비트 아나콘다를 설치하고, 가상 환경을 두 개(32비트, 64비트) 설정해서 사용해야 함.

​

설정 방법은 맨 아래에....

​

<고려해야 할 점 1> 

키움 Open API 는 32비트 아나콘다를 설치해야만 사용할 수 있음

64비트 아나콘다를 설치 시, "QAxBase::setControl: requested control KHOPENAPI.KHOpenAPICtrl.1 could not be instantiated" 와 같은 에러가 발생함

https://wikidocs.net/2825


 
위키독스
온라인 책을 제작 공유하는 플랫폼 서비스

wikidocs.net

<고려해야 할 점 2>

케라스는 64비트 아나콘다를 설치해야만 사용할 수 있음

<고려해야 할 점 3>

텐서플로 사용 시 파이썬 3.7이 내장된 아나콘다 설치하면, 텐서플로를 못씀.. (2018년 12월 현재)

 -> 파이썬 3.6이 내장된 아나콘다를 설치하면 됨..

​

download the most recent Anaconda installer that included Python 3.5 (Anaconda 4.2.0) or Python 3.6 (Anaconda 5.2.0). You can download either of these from our archive. Scroll down the page until you find the version you need for your platform.

http://docs.anaconda.com/anaconda/user-guide/faq/#how-do-i-get-the-latest-anaconda-with-python-3-5

 
Anaconda installer archive
Filename Size Last Modified MD5 Anaconda2-5.3.1-Linux-x86.sh 507.6M 2018-11-19 13:37:35 5685ac1d4a14c4c254cbafc612c77e77 Anaconda2-5.3.1-Linux-x86_64.sh 617.8M 2018-11-19 13:37:31 4da47b83b1eeac1ca8df0a43f6f580c8 Anaconda2-5.3.1-MacOSX-x86_64.pkg 628.4M 2018-11-19 13:37:38 d6139f371aa6cf81c3f002ecdd

repo.anaconda.com

<설정 방법> 

1. 아래에서 Anaconda3-5.2.0-Windows-x86_64.exe 를 받아서 설치함

https://repo.anaconda.com/archive/

 
Anaconda installer archive
Filename Size Last Modified MD5 Anaconda2-5.3.1-Linux-x86.sh 507.6M 2018-11-19 13:37:35 5685ac1d4a14c4c254cbafc612c77e77 Anaconda2-5.3.1-Linux-x86_64.sh 617.8M 2018-11-19 13:37:31 4da47b83b1eeac1ca8df0a43f6f580c8 Anaconda2-5.3.1-MacOSX-x86_64.pkg 628.4M 2018-11-19 13:37:38 d6139f371aa6cf81c3f002ecdd

repo.anaconda.com

​

2. Anaconda Prompt 를 실행한 후, 32비트 가상환경을 만들어줌

 1) 32비트 가상환경 생성

set CONDA_FORCE_32BIT=1
conda create -n py36_32 python=3.6.5
※ 어떤 버전의 아나콘다를 설치했는 지에 따라 파이썬 버전은 다를 수있으므로, 가상환경을 생성하기 전에 아래 명령어로 파이썬 버전을 미리 확인해야 함

​

# python --version

​

​

 2) 32비트 가상환경 활성화 방법

set CONDA_FORCE_32BIT=1
activate py36_32
 3) 32비트 가상환경 비활성화 방법

deactivate py36_32
3. 마찬가지 방법으로, 64비트 가상환경을 만들어줌

 1) 64비트 가상환경 생성

set CONDA_FORCE_32BIT=
conda create -n py36_64 python=3.6.5
 2) 64비트 가상환경 활성화 방법

set CONDA_FORCE_32BIT=
activate py36_64
 3) 64비트 가상환경 비활성화 방법

deactivate py36_64
4. 키움증권 Open API 를 사용하는 파이썬 파일은 32비트 환경에서 실행시키고, 케라스를 사용하는 파이썬 파일은 64비트 환경에서 실행시키면 잘 돌아감
