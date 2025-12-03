# 항공기 착륙 위험 정도 예측 모델 (LRI Engine)
## 2025 스페이스 해커톤 Aerosafers팀 웹사이트

초기 화면

<img width="549" height="289" alt="image" src="https://github.com/user-attachments/assets/d31e6cd8-c15d-43b0-b25f-4a9cff0510a7" />

* 노란 영역은 ‘인천 FIR 공역’으로 ‘KASS’를 비롯한 ‘LRI Engine 서비스 커버리지’ 

이 프로그램은 LRI(Landing Risk Index) 모델의 미리 정의된 파라미터와 수식들을 적용하여, Node.js 기반 프런트엔드와 Python Uvicorn 백엔드를 내부 코드로 연계한다.

실제 위성 데이터(NetCDF, RINEX)를 파싱하는 복잡한 로직 대신, LRI 계산에 필요한 파라미터를 위도 경도 등등의 간단한 입력값으로 대체하여 LRI 산출 로직과 기술 스택 연동 구조를 시연하는 데 중점을 둔다. 
지도 활용은 gpd GeoJSON을 작업한 leaflet 기반으로 실행한다.

## 가능한 상호 작용:

## 위도 경도 입력
지도상의 위치 (longitude, latitude) 값을 입력할 수 있다.
## 비행체 종류 선택

<img width="435" height="242" alt="image" src="https://github.com/user-attachments/assets/f5a1f762-e8b9-46d3-9e15-f79eebf1dab1" />

선택 가능한 UAM 이름: CTOL, STOL, VTOL, eVTOL, eCTOL, eSTOL
## 비행 기체 구분 선택

<img width="437" height="144" alt="image" src="https://github.com/user-attachments/assets/4ef2aab3-1679-47ac-9cfa-09b462876bd4" />

선택 가능한 구분: 고정익, 회전익
## 분석 실행 시
### Best case scenario

착륙 위험 정도: 미미/없음 aka Very Good

<img width="549" height="289" alt="image" src="https://github.com/user-attachments/assets/e592ad61-b3de-4e39-aee8-6c796d692b81" />


### Worst case scenarios 
착륙 위험 정도: 착륙 불가 aka Hard Stop

시나리오 A: 구름으로 인한 시경/기상 악화 상황

![시나리오 A](https://github.com/user-attachments/assets/47215941-abe3-4d12-b683-b956d5b08663)


시나리오 B: 항법 무결성/오류로 인한 상황

![시나리오 B](https://github.com/user-attachments/assets/70d78c9d-c9f3-4211-b92f-46dee81a4d14)

시나리오 C: 인근 산악/구조물이 있는 상황  

![시나리오 C](https://github.com/user-attachments/assets/2e13e24c-bc96-4fa5-8e34-b6d1412e0ce8)

### 확인 가능한 값:
분석결과: 
맵 위치에서 선택 지정한 위도와 경도에 비행기가 표시된 것을 확인 가능. 

최종등급: 
LRI 시나리오 명칭과 그 수치를 확인 가능. 

전달받은 3대 위험 요소 점수:
천리안 날씨 정보: Weather, KASS 정보: Navigation, 아리랑 정보: Terrain 의 값들을 확인 가능


근거(위성데이터 융합):최종 등급을 결정 지은 하위 요인들을 총 3개 수치화하여 시각적으로 표시한다.

<img width="422" height="856" alt="image" src="https://github.com/user-attachments/assets/517d7b02-6fa2-4600-861f-0778f5e314db" />

하위 항목: 

가시성값: 구름 감쇠계수로 인한 감점의 정도를 나타냄.
예) W-Score: 55.76, 구름 감쇠: 0.44, 시정: 37.8

항법값: HPL VPL값으로 일어나는 항법 무결성 정도를 나타냄. 
예) N-Score: 100, HPL: 16.7m, VPL: 14.8m

지형값: 장애물 비율을 나타냄. 
예) T-Score: 19.71, 지형 복잡도: 0.01

## 로컬 환경 설치 방법
우분투 Ubuntu 기반 테스트 메뉴얼임을 유의.

### Python venv 설치 install python if missing
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

### Venv 환경 추가 및 설정하기 create and activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 백엔드 요구조건 일괄 설치 install python dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 노드 패키지 모두 업데이트 install node dev dependencies
```bash
sudo apt install nodejs npm -y
npm install
```

## 테스트 확인 방법

### 프론트엔드만 실행 run frontend only
```bash
npm run ㄴstart
```

### 백엔드만 실행 run backend only
```bash
npm run backend
<verify it is running>
curl -v http://127.0.0.1:8000/api/map
```

### (최종 사이트 환경) 동시에 실행 run frontend + backend concurrently
```bash
npm run dev
```

### Chack: Is both frontend and backend properly executed?
```bash
[frontend]   http://172.27.105.189:3000
[frontend] Hit CTRL-C to stop the server
[frontend] Open: http://127.0.0.1:3000
[backend] INFO:     Will watch for changes in these directories: ['/mnt/d/usage/noacademics/space hackathon/aerosafers']
[backend] INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[backend] INFO:     Started reloader process [2502] using StatReload
```
Must see this line to be functional
### 로컬 환경 사이트 연결 해제 send Termination of server connection
after ctrl+c

```bash
[backend] Shutting down backend server gracefully...
[frontend] npm run frontend exited with code SIGINT
```
모두 vscode 콘솔에서 확인되면 해제 완료! Website port terminated successfully


