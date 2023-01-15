[docker command]

## docker run

1. docker run [OPTIONS] IMAGE [command] [arg...]
### --add-host : 사용자 지정 호스트-IP 매핑 추가(host:ip)
### -a : --attach STDIN, STDOUT or STDERR 에 연결
### -d : --detach 백그라운드에서 컨테이너 실행 및 컨테이너 ID 출력
### --detach-keys : 컨테이너 분리를 위한 키 시퀀스 재정의
### -e : --env 환경 변수 설정
### --env-file : 환경 변수 파일에서 읽기
### --expose : 포트 또는 포트 범위 노출
### --group-add : 가입할 그룹 추가
### -h : --hostname 컨테이너 호스트 이름
### --mount : 파일 시스템 마운트 컨테이너에 연결
### --name : container 에 이름 할당
### --net : 컨테이너를 네트워크에 연결
### --net-alias : 컨테이너에 대한 네트워크 범위 별칭 추가
### --network : 컨테이너를 네트워크에 연결
### --network-alias : 컨테이너에 대한 네트워크 범위 별칭 추가
### --pull : 실행하기 전에 이미지 가져오기
### --read-only : 컨테이너의 루트 파일 시스템을 읽기 전용으로 마운트
### --restart : 컨테이너 종료시 적용할 재시작 정책
### --rm : 컨테이너가 종료되면 자동으로 제거
### -v : --volume 불륨 마운트 바인딩
### --volume-driver : 컨테이너용 옵션 불륨 드라이버
### --volumes-from : 지정된 컨테이너에서 볼륨 마운트
### -w : --workdir 컨테이너 내부의 작업 디렉토리

## docker image
### docker iamge build : Dockerfile 에서 이미지 빌드
### docker image history : 이미지의 history 표시
### docker iamge ls : 이미지 나열
### docker iamge prune : 사용하지 않는 이미지 제거
### docker iamge pull : 레지스트리에서 이미지 또는 저장소 가져오기
### docker iamge push : 이미지 또는 저장소를 레지스트리에 push 
### docker iamge rm : 하나 이상의 이미지 제거
### docker iamge tag : source_image를 참조하는 target_image tag 생성
### docker ps : 

## docker container 
1. docker container [COMMAND]
### docker container attach : 실행 중인 컨테이너에 로컬 표준 입력,출력 & 오류 스트림 연결
### docker container commit : 컨테이너의 변경 사항에서 새 이미지 만들기
### docker container cp : 컨테이너와 로컬 파일 시스템 간에 파일/폴더 복사
### docker container create : 새 컨테이너 만들기
### docker container exec : 실행 중인 컨테이너에서 명령 실행
### docker container inspect : 하나 이상의 컨테이너에 대한 자세한 정보 표시
### docker container kill : 하나 이상의 실행 중인 컨테이너 종료
### docker container logs : 컨테이너의 로그 가져오기
### docker container ls : 컨테이너 나열
### docker container pause : 하나 이상의 컨테이너 내 모든 프로세스 일시 중지
### docker container port : 컨테이너에 대한 포트 매핑 또는 특정 매핑 나열
### docker container prune : 중지된 모든 컨테이너 제거
### docker container rename : 컨테이너 이름 바꾸기
### docker container restart : 컨테이너 다시 시작
### docker container rm : 컨테이너 제거
### docker container start : 컨테이너 시작
### docker container run : 새 컨테이너에서 명령 실행
### docker container stop : 실행중인 컨테이너 중지
### docker container top : 컨테이너의 실행 중인 프로세스 표시
### docker container unpause : 컨테이너 내의 모든 프로세스 일시 중지 해제

## docker compose 
1. docker compose [COMMAND]
### docker-compose up -d : 컨테이너 생성 및 시작
### docker-compose scale web =10 : 생성 컨테이너 수
### docker-compose ps : 컨테이너 목록
### docker-compose logs : 컨테이너 로그
### docker-compose run web /bin/cal : 시작 + 명령 실행
### docker-compose exec web bash : 명령 실행
### docker-compose start : 전체 컨테이너 시작
### docker-compose restart : 전체 컨테이너 다시 시작
### docker-compose kill : 전체 컨테이너 강제 정지
### docker-compose rm : 전체 컨테이너 삭제
### docker-compose build --no-cache : 전체 컨테이너를 빌드



