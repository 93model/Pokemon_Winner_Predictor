# Pokemon_Winner_Predictor


## 🛠 서비스 개요
포켓몬스터 게임을 하다 보면  NPC의 포켓몬에 지기도 한다.
포켓몬 타입만 18개, 두번째 타입을 가진 포켓몬의 경우의 수 까지 합하면 총 (18*17-16)=290 종류의 타입이 존재
갈수록 복잡해져 가는 포켓몬배틀 환경
포켓몬스터 6세대 X,Y 약 721마리의 포켓몬이 대상

### -> 포켓몬 배틀의 대략적인 결과를 예측한다

## ❓ 데이터

포켓몬 6세대 X,Y 721 기준
메가 진화, 다른 폼의 포켓몬을 포함하여 
총 800 마리의 포켓몬 Pokemon with stats(https://www.kaggle.com/datasets/abcsds/pokemon)

kaggle Pokemon- Weedle's Cave(https://www.kaggle.com/datasets/terminus7/pokemon-challenge)
에서 제공하는 포켓몬 승리, 패배 데이터 5만건

## 🧹 데이터 전처리



SQLite 와 DBeaver를 사용하야 데이터를 정리후 한글화

Flask를 통해 포켓몬 두 마리의 이름을 입력하면 어떤 포켓몬이 이길 확률이 높은지 나오는 API를 개발

## 🛠 한계

## ✔️ API 

## 🔍 Metabase 시각화 

