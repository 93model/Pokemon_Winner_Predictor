# poke_winner_go

## MENTAL  ILLNESS Machine Learning Analysis

## 🛠 서비스 개요
포켓몬스터 게임을 하다 보면  NPC의 포켓몬에 지기도 한다.
포켓몬 타입만 18개, 두번째 타입을 가진 포켓몬의 경우의 수 까지 합하면 총 (18*17-16)=290 종류의 타입이 존재
갈수록 복잡해져 가는 배틀 환경
포켓몬스터 6세대 X,Y 약 721마리의 포켓몬이 대상

### 포켓몬 배틀의 대략적인 결과를 예측한다

## ❓ 데이터

포켓몬 6세대 X,Y 721 기준
메가 진화, 다른 폼의 포켓몬을 포함하여 총 800 마리의 포켓몬 Pokemon with stats(https://www.kaggle.com/datasets/abcsds/pokemon)
kaggle Pokemon- Weedle's Cave(https://www.kaggle.com/datasets/terminus7/pokemon-challenge)에서 제공하는 포켓몬 승리, 패배 데이터 5만건

## 🧹 데이터 전처리
SQLite 와 DBeaver를 사용하야 데이터를 정리후 한글화
Flask를 통해 포켓몬 두 마리의 이름을 입력하면 어떤 포켓몬이 이길 확률이 높은지 나오는 API를 개발

## 🛠 한계
데이터의 절대적 수가 부족 OSMI Mental Health in Tech Survey : 약 1200개 
미국 중심 : 정신 질환에 대한 개방 정도가 한국과 다름
스스로 실시한 설문조사 : 정신 질환에 관심이 있어서  찾아온 사람들, 치료 비율 높음

## ✔️ 결과
머신러닝 방식 적용 !
RandomForest  + RandomizedSearchCV 사용
하이퍼 파라미터의 최적값 적용. 
적은 데이터의 개수를 극복하기 위해  n_iter=20,  cv=10,사용
최적의 임계값 찾기 idx: 34 , threshold: 0.5542227788687464

![캡처](/img/f1_score.png)

## 🔍 결과 
관측치를 예측하는 특성
![캡처](/img/feature_importance.png)
 
family_history (가족력)
care_options ( 회사에서 제공하는 정신 건강 관리 옵션 )
