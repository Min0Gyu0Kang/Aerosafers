<img width="1003" height="710" alt="image" src="https://github.com/user-attachments/assets/cef0a540-6693-4020-8502-9f8bc87c3b42" /># LRI 기반 모의 테스트(Mock Test)
## 2025 스페이스 해커톤 Aerosafers팀 모의실험용 웹사이트

초기 화면
<img width="1842" height="862" alt="image" src="https://github.com/user-attachments/assets/6e190259-0b5e-49e0-a210-e8d279f89639" />

실행 후
<img width="723" height="795" alt="image" src="https://github.com/user-attachments/assets/b5b25dc0-37cc-40a9-95f7-0597fd89f5c5" />


이 프로그램은 LRI(Landing Risk Index) 모델의 미리 정의된 파라미터와 수식들을 적용하여, Node.js 기반 프런트엔드와 Python Uvicorn 백엔드를 내부 코드로 연계한다.

실제 위성 데이터(NetCDF, RINEX)를 파싱하는 복잡한 로직 대신, LRI 계산에 필요한 파라미터를 임의의 저장한 데이터 (Preset)로 대체하여 LRI 산출 로직과 기술 스택 연동 구조를 시연하는 데 중점을 둔다. 

맵 영역이 비워져 있는 이유는 추가 구현시 python anaconda venv 기반 고도화된 환경으로 GeoPandas Mapbox/Leaflet 구축이 추가되어야 하기 때문에 해당 버전은 간단한 실험부터 진행함.

## 가능한 상호 작용:

## LRI 지수 선택

<img width="229" height="186" alt="image" src="https://github.com/user-attachments/assets/b1060f0c-3b54-4149-88f4-a77dae56e492" />

Scenario 옆의 드롭다운 막대를 선택하면 내부의 random, very_good, severe, warning, hard_stop, random 선택지 중 택 1을 할 수 있다. 
### 시나리오 설명: 

random: default와 동일, 데이터 값도 무작위로 전달받음.

very_good: 항공기 착륙 가능인 경우. 초록으로 표시.

severe: 항공기 착륙에 충돌이 많이 우려될 경우. 빨강으로 표시.

warning: 항공기 착륙에 난항이 예상되는 경우. 노랑으로 표시.

hard_stop: 항공기 착륙 불가인 경우. 색이 페이드 인/아웃하는 빨강 계열로 표시.

## 맵 영역 클릭 시 (지도 없는 상태로 테스트)

<img width="875" height="862" alt="스크린샷 2025-11-10 223653" src="https://github.com/user-attachments/assets/ef9f891a-eae1-45f1-9b1e-35d758a460f9" />

<img width="888" height="823" alt="스크린샷 2025-11-10 223746" src="https://github.com/user-attachments/assets/f3b91ee5-8311-4448-9dfe-def7e0e42c71" />

<img width="849" height="837" alt="스크린샷 2025-11-10 223755" src="https://github.com/user-attachments/assets/e75d4a1a-c82b-4a73-92ce-a874187cb3b2" />

<img width="1011" height="835" alt="스크린샷 2025-11-10 223806" src="https://github.com/user-attachments/assets/b300f876-fce7-4a45-9446-2c09a48327d8" />

초기 화면의 LRI 계산 결과가 표시됩니다 영역이 결과값으로 채워짐.

### 확인 가능한 값:
분석결과: 
맵 위치에서 선택 지정한 위도와 경도를 확인 가능. (모의에서는 임의 데이터 값들만 확인이 가능)

최종등급: 
LRI 시나리오 명칭과 그 수치를 확인 가능. 

전달받은 3대 위험 요소 점수:
가시성 점수 V, 항법 무결성 점수 N, 표면 지형 점수 S 의 값들을 확인 가능


근거(위성데이터 융합):최종 등급을 결정 지은 하위 요인들을 총 3개 수치화하여 시각적으로 표시한다.

<img width="535" height="247" alt="image" src="https://github.com/user-attachments/assets/e4e6d542-d16f-4981-b317-d171aca285ee" />

하위 항목: 

가시성값: 구름 감쇠계수로 인한 감점의 정도를 나타냄.

항법값: HPL값으로 일어나는 항법 무결성 정도를 나타냄.

- 결과 1: 유지 상태
- 결과 2: 저하 상태 

지형값: 장애물 비율을 나타냄. 추가로 표면 습윤/결빙 위험값을 나타냄. 

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
npm run frontend
```

### 백엔드만 실행 run backend only
```bash
npm run backend
```

### 백엔드: 위험지수별 테스트 (scenario curl testing)

#### 위도 경도 시나리오 값 넘겨주기 change lat, lon, and scenario names for checking. Examples below

```bash
curl -s -X POST http://127.0.0.1:8000/api/calculate_lri \
    -H "Content-Type: application/json" \
    -d '{"lat":37.5,"lon":127.0,"scenario":"severe"}'

curl -s -X POST http://127.0.0.1:8000/api/calculate_lri \
    -H "Content-Type: application/json" \
    -d '{"lat":37.5,"lon":127.0,"scenario":"warning"}'

curl -s -X POST http://127.0.0.1:8000/api/calculate_lri \
    -H "Content-Type: application/json" \
    -d '{"lat":37.5,"lon":127.0,"scenario":"hard_stop"}'

curl -s -X POST http://127.0.0.1:8000/api/calculate_lri \
    -H "Content-Type: application/json" \
    -d '{"lat":37.5,"lon":127.0,"scenario":"very_good"}'

```

### (최종 사이트 환경) 동시에 실행 run frontend + backend concurrently
```bash
npm run dev
```

### 로컬 환경 사이트 연결 해제 send Termination of server connection
after ctrl+c

```bash
[frontend] npm run frontend exited with code SIGINT
[backend]
[backend] Shutting down backend server gracefully...
[backend] INFO:     Stopping reloader process [2956]
[backend] npm run backend exited with code SIGINT
```
모두 vscode 콘솔에서 확인되면 완료! Website port terminated successfully


