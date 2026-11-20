# 자연어 처리 학습 자료

형태소 분석과 단어 표현에서 시작해 텍스트 유사도, 감정 분류, 개체명 인식, LLM과 Transformer 응용을 살펴봅니다.

## 학습 순서

1. [형태소 분석](01_%ED%86%A0%ED%81%AC%EB%82%98%EC%9D%B4%EC%A7%95_%ED%98%95%ED%83%9C%EC%86%8C%EB%B6%84%EC%84%9D.ipynb)
2. [원핫 벡터](02_%EC%9E%84%EB%B2%A0%EB%94%A9_%EC%9B%90%ED%95%AB%EC%9D%B8%EC%BD%94%EB%94%A9.ipynb)
3. [Word2Vec](03_Word2Vec_%EB%8B%A8%EC%96%B4%EC%9E%84%EB%B2%A0%EB%94%A9.ipynb)
4. [n-gram·코사인 유사도](04_%ED%85%8D%EC%8A%A4%ED%8A%B8%EC%9C%A0%EC%82%AC%EB%8F%84_%EC%95%A4%EA%B7%B8%EB%9E%A8%EA%B3%BC%EC%BD%94%EC%82%AC%EC%9D%B8.ipynb)
5. [CNN 감정 분류](05_%EA%B0%90%EC%A0%95%EB%B6%84%EB%A5%98_CNN.ipynb)
6. [BiLSTM 개체명 인식](06_%EA%B0%9C%EC%B2%B4%EB%AA%85%EC%9D%B8%EC%8B%9D_%EC%96%91%EB%B0%A9%ED%96%A5LSTM.ipynb)
7. [LLM·RAG 개요](07_%EB%8C%80%EA%B7%9C%EB%AA%A8%EC%96%B8%EC%96%B4%EB%AA%A8%EB%8D%B8_%EA%B0%9C%EB%85%90.ipynb)
8. [전이학습 개요](08_%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5_%EA%B0%9C%EB%85%90.ipynb)
9. [BPE·WordPiece](09_%ED%86%A0%ED%81%B0%ED%99%94_BPE%EC%99%80WordPiece.ipynb)
10. [Transformer 어텐션](10_%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8_%EC%96%B4%ED%85%90%EC%85%98.ipynb)
11. [IMDb·NSMC BERT 감성 분류](11_%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5_IMDb%EC%99%80NSMC%EA%B0%90%EC%84%B1%EB%B6%84%EB%A5%98.ipynb)
12. [GPT·GPT-2 텍스트 생성](12_GPT_%ED%85%8D%EC%8A%A4%ED%8A%B8%EC%83%9D%EC%84%B1.ipynb)

## 실행 환경과 경로

Jupyter 커널의 작업 디렉터리를 `06_NLP/`로 지정하세요. Notebook의 상대 경로는 이 디렉터리를 기준으로 합니다.

검증 환경은 Python 3.13.15, NumPy 2.5.2, Pandas 3.0.5, TensorFlow 2.20.0, Keras 3.15.1, PyTorch 2.13.0, Transformers 5.15.1, Gensim 4.4.0, KoNLPy 0.6.0, Korpora 0.2.0, Keras NLP 0.31.1입니다. 설치 버전 목록이며 모든 실습의 호환성을 보장하지 않습니다.

- 1번의 Kkma·Komoran 실행에는 호환 Java 런타임이 필요합니다. 검증 환경에서는 Java 런타임이 확인되지 않았습니다. 로컬 원본의 Oracle JDK ZIP은 Windows용입니다.
- 3번의 영문 Word2Vec은 `data/text8.txt`를 사용하고, NSMC 실습은 `data/nsmc_tokenizing.csv`를 읽습니다. 1번의 CSV 출력 위치는 `06_NLP/` 바로 아래이므로 두 경로를 구분하세요.
- 9번은 Korpora에서 NSMC를 내려받고 `data/nsmc/train.txt`, `data/nsmc/test.txt`를 생성합니다. 토크나이저를 저장·로드하려면 선행 실행과 `bpe/`, `wordpiece/`가 필요합니다.
- 11번의 IMDb 로더는 이미 압축이 풀린 `data/aclImdb/`를 사용합니다. 기존 Colab 설치·압축 해제 셀은 로컬 검증 범위에서 제외했습니다. BERT 사전학습 모델·평가지표, 12번의 GPT 계열 모델은 별도 다운로드나 캐시가 필요할 수 있습니다. 11번 학습 실행은 체크포인트를 `bert_kor_nsmc/`에 생성합니다.
- 대용량 학습, 추론, 외부 모델 로드, 토크나이저 저장은 이번 CPU 격리 검증 범위에 포함되지 않았습니다.

## Phase 2 실행 검증

전체 완료 3개, 부분 완료 6개, 실패 1개, 실행 불필요 2개, 실행 불가 0개입니다. 로컬 데이터 읽기와 안전한 연산을 원래 순서대로 격리 실행했습니다. 네트워크와 파일 쓰기를 차단했으며, 결과는 검증한 셀에 한합니다. 원본 저장 출력은 보존했습니다. 실패는 해당 셀의 코드 또는 검증 환경 제약에 한해 해석해야 합니다.

| 차시 | 상태 | 성공 / 실패 / 제외 |
| --- | --- | ---: |
| 01 | 부분 완료 | 5 / 0 / 6 |
| 02 | 전체 완료 | 6 / 0 / 0 |
| 03 | 부분 완료 | 4 / 0 / 16 |
| 04 | 전체 완료 | 23 / 0 / 0 |
| 05 | 부분 완료 | 23 / 0 / 15 |
| 06 | 부분 완료 | 24 / 0 / 14 |
| 07 | 실행 불필요 | 0 / 0 / 0 |
| 08 | 실행 불필요 | 0 / 0 / 0 |
| 09 | 부분 완료 | 3 / 0 / 20 |
| 10 | 전체 완료 | 30 / 0 / 0 |
| 11 | 부분 완료 | 9 / 0 / 36 |
| 12 | 실패 | 0 / 1 / 6 |


## GitHub 공개 범위와 자료 준비

공개 후보는 Notebook 12개와 이 README입니다. 재배포 권한을 확인하기 전에는 IMDb, NSMC, NER, ChatbotData, text8 등 원본 데이터, 강의 이미지, Oracle JDK ZIP을 공개본에 포함하지 않습니다. 모델·토크나이저·체크포인트 생성물도 공개 범위에서 제외합니다. 데이터와 모델이 필요한 경우 출처와 이용 조건을 확인한 뒤 해당 상대 경로에 준비하세요. 이미지가 빠진 경우 GitHub의 Notebook 미리보기에 일부 그림이 표시되지 않을 수 있습니다.

IMDb 데이터의 로컬 README에 있는 인용 정보는 출처 확인에 참고할 수 있습니다. 리뷰·말뭉치의 개인정보 후보와 재배포 권한은 별도 확인이 필요합니다. IMDb 학습·평가 분할에 완전히 동일한 리뷰가 있어 성능 비교 시 이 점을 고려해야 합니다.
