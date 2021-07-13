# 뉴스 토픽 분류 경진대회(텍스트 분류)
## 1. 데이터
### 1개 문장 형태의 텍스트 - 정수 인덱스 형태의 라벨 데이터로 구성
### 한자, 영어, 외래어 포함된 뉴스 제목, IT과학, 정치, 경제, 스포츠, 생활문화, 세계로 구성됨

## 2. 접근
### 전처리
- konlpy 라이브러리 Kkma를 이용하여 전처리 진행 : 영어, 한자를 포함한 명사만 추출
- 토큰화: 케라스 토크나이저 이용

### 모델
- embedding layer + bidirectional LSTM stacked structure
- Attention Layer(Encoder) + LSTM
- Transformer(Encoder) 
- BERT model from tensorflow
