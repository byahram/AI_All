# 딥러닝 학습 자료

신경망의 기본 구조에서 시작해 이미지 분류, 전이학습, 자연어 처리, Transformer·BERT와 모델 해석까지 학습한 기록입니다. 같은 주제의 여러 실습은 각 Notebook의 번호순으로 살펴볼 수 있습니다.

## 실행 환경

Phase 3-B 검증에 사용한 주요 환경은 Python 3.13.15, TensorFlow 2.20.0, Keras 3.15.1, PyTorch 2.13.0, NumPy 2.5.2, Pandas 3.0.5입니다. 이는 검증 환경의 기록이며 전체 실습의 호환 버전을 보장하지 않습니다. 저장소 루트의 [requirements.txt](../requirements.txt)는 버전이 고정되어 있지 않습니다.

Jupyter의 커널과 작업 디렉터리를 확인하세요. Notebook의 `./data/`, `./image/`, `./model/` 등 상대 경로는 작업 디렉터리가 `05_DL/`일 때를 기준으로 합니다.

| 위치 | 용도 |
| --- | --- |
| `data/` | 텍스트·표 데이터, `nsmc/`, `ner/`, `catanddog/` 실습 데이터 |
| `image/` | 딥러닝·이미지 분류·NLP·LLM·PyTorch 설명 그림 |
| `best-cnn-model.keras` | 12번·22번의 셀 3에서 참조하는 CNN 가중치 포함 모델 |
| `model/`, `checkpoints/` | 학습을 실행할 때 필요한 모델 및 체크포인트 출력 위치 |
| `data/nsmc_korpora/`, `bpe/`, `wordpiece/` | 일부 토큰화 실습의 전처리 데이터 및 토크나이저 산출물 위치 |

### 다운로드와 선행 학습

- Fashion-MNIST, MNIST, CIFAR-10, IMDB, Reuters, Korpora 데이터나 사전학습 가중치 및 Hugging Face 모델을 사용하는 실습은 로컬 캐시가 없으면 다운로드가 필요할 수 있습니다.
- 일부 PyTorch 실습은 CUDA 장치를 직접 지정합니다. 해당 셀을 원래 설정으로 실행하려면 호환되는 GPU·CUDA 환경이 필요합니다. 형태소 분석 실습의 KoNLPy 분석기는 Java 런타임을 요구할 수 있습니다.
- 모델을 다시 생성하려면 해당 실습의 저장 경로와 상위 디렉터리를 확인한 뒤 학습·저장 셀을 로드 셀보다 먼저 실행하세요. 기존 모델을 덮어쓰지 않도록 별도 작업 복사본에서 실행하세요. 12번·22번은 `./best-cnn-model.keras`를 읽고, CNN 학습 코드 중 일부는 `./model/best-cnn-model.keras`에 저장합니다.
- 40번 셀 28은 Gensim의 `Text8Corpus`로 `./data/text8.txt`를 읽습니다. 공개본에 이 파일이 없다면 출처와 이용·배포 조건이 확인된 text8 코퍼스를 같은 위치에 준비해야 해당 실습과 후속 의존 셀을 실행할 수 있습니다. 기존 Notebook에는 공식 제공처를 확인할 링크가 없습니다.

### 실행 검증 범위

Phase 3-B 결과는 전체 검증 완료 3개, 부분 검증 완료 29개, 실행 실패 8개, 실행 불필요 5개, 실행 불가 4개입니다. 격리된 CPU 환경에서 안전한 셀을 검증했으며, 학습·모델 로드·다운로드·GPU 사용은 제외했습니다. Java·KoNLPy 및 Transformers 격리 제한이 남아 있습니다. CNN 파일은 컨테이너와 입력 형태의 정적 검증만 완료했습니다. 저장 출력과 동일한 결과가 재현되는지는 실습별 환경과 선행 단계에 따라 달라집니다.

### GitHub 공개 범위와 실습 자료

GitHub 공개본에는 Notebook 49개가 포함됩니다. 재배포 권한 확인이 필요한 이미지·데이터·모델은 공개본에서 제외됩니다. 해당 이미지가 없으면 GitHub의 Notebook 미리보기에 그림이 표시되지 않을 수 있습니다.

실습은 `05_DL/`을 작업 디렉터리로 사용하세요. 필요한 데이터와 모델은 출처와 이용 조건을 확인한 뒤 Notebook에 지정된 상대 경로에 준비하세요. 해당 입력이나 선행 체크포인트가 없으면 그 자료를 사용하는 셀과 후속 셀의 실행이 제한됩니다.

위 Phase 3-B 결과는 당시 로컬 검증 환경을 기준으로 합니다. 공개본의 실행 범위는 자료 준비 여부, Java·KoNLPy와 GPU 환경, 외부 다운로드 및 선행 체크포인트에 따라 달라집니다. 두 CNN 시각화 실습에서 사용하는 모델은 정적 검증만 완료했습니다.

## Notebook 학습 순서

### 신경망 기초·훈련과 분류

1. [단층신경망 패션분류](01_%EB%8B%A8%EC%B8%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%8C%A8%EC%85%98%EB%B6%84%EB%A5%98.ipynb)
2. [퍼셉트론 구조와동작](02_%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0_%EA%B5%AC%EC%A1%B0%EC%99%80%EB%8F%99%EC%9E%91.ipynb)
3. [다층신경망 은닉층과활성화](03_%EB%8B%A4%EC%B8%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D_%EC%9D%80%EB%8B%89%EC%B8%B5%EA%B3%BC%ED%99%9C%EC%84%B1%ED%99%94.ipynb)
4. [단층신경망 패션예측실습](04_%EB%8B%A8%EC%B8%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%8C%A8%EC%85%98%EC%98%88%EC%B8%A1%EC%8B%A4%EC%8A%B5.ipynb)
5. [다층신경망 구성과최적화](05_%EB%8B%A4%EC%B8%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D_%EA%B5%AC%EC%84%B1%EA%B3%BC%EC%B5%9C%EC%A0%81%ED%99%94.ipynb)
6. [훈련도구 손실과모델저장](06_%ED%9B%88%EB%A0%A8%EB%8F%84%EA%B5%AC_%EC%86%90%EC%8B%A4%EA%B3%BC%EB%AA%A8%EB%8D%B8%EC%A0%80%EC%9E%A5.ipynb)
7. [훈련도구 검증손실과조기종료](07_%ED%9B%88%EB%A0%A8%EB%8F%84%EA%B5%AC_%EA%B2%80%EC%A6%9D%EC%86%90%EC%8B%A4%EA%B3%BC%EC%A1%B0%EA%B8%B0%EC%A2%85%EB%A3%8C.ipynb)
8. [합성곱신경망 구성기초](08_%ED%95%A9%EC%84%B1%EA%B3%B1%EC%8B%A0%EA%B2%BD%EB%A7%9D_%EA%B5%AC%EC%84%B1%EA%B8%B0%EC%B4%88.ipynb)
9. [붓꽃 다중분류](09_%EB%B6%93%EA%BD%83_%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98.ipynb)
10. [합성곱신경망 패션이미지분류](10_%ED%95%A9%EC%84%B1%EA%B3%B1%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%8C%A8%EC%85%98%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98.ipynb)
11. [초음파 광물예측](11_%EC%B4%88%EC%9D%8C%ED%8C%8C_%EA%B4%91%EB%AC%BC%EC%98%88%EC%B8%A1.ipynb)
12. [합성곱신경망 필터시각화](12_%ED%95%A9%EC%84%B1%EA%B3%B1%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%84%ED%84%B0%EC%8B%9C%EA%B0%81%ED%99%94.ipynb)
13. [보스턴 주택가격회귀](13_%EB%B3%B4%EC%8A%A4%ED%84%B4_%EC%A3%BC%ED%83%9D%EA%B0%80%EA%B2%A9%ED%9A%8C%EA%B7%80.ipynb)

### 순환신경망·CNN·텍스트 분류 실습

14. [순환신경망 순차데이터기초](14_%EC%88%9C%ED%99%98%EC%8B%A0%EA%B2%BD%EB%A7%9D_%EC%88%9C%EC%B0%A8%EB%8D%B0%EC%9D%B4%ED%84%B0%EA%B8%B0%EC%B4%88.ipynb)
15. [순환신경망 IMDB분류실습](15_%EC%88%9C%ED%99%98%EC%8B%A0%EA%B2%BD%EB%A7%9D_IMDB%EB%B6%84%EB%A5%98%EC%8B%A4%EC%8A%B5.ipynb)
16. [순환신경망 IMDB분류변형](16_%EC%88%9C%ED%99%98%EC%8B%A0%EA%B2%BD%EB%A7%9D_IMDB%EB%B6%84%EB%A5%98%EB%B3%80%ED%98%95.ipynb)
17. [합성곱신경망 패딩과풀링](17_%ED%95%A9%EC%84%B1%EA%B3%B1%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%8C%A8%EB%94%A9%EA%B3%BC%ED%92%80%EB%A7%81.ipynb)
18. [순환신경망 LSTM과GRU기초](18_%EC%88%9C%ED%99%98%EC%8B%A0%EA%B2%BD%EB%A7%9D_LSTM%EA%B3%BCGRU%EA%B8%B0%EC%B4%88.ipynb)
19. [합성곱신경망 패션분류검증](19_%ED%95%A9%EC%84%B1%EA%B3%B1%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%8C%A8%EC%85%98%EB%B6%84%EB%A5%98%EA%B2%80%EC%A6%9D.ipynb)
20. [다층신경망 파이토치학습](20_%EB%8B%A4%EC%B8%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98%ED%95%99%EC%8A%B5.ipynb)
21. [파이토치 패션분류기초](21_%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98_%ED%8C%A8%EC%85%98%EB%B6%84%EB%A5%98%EA%B8%B0%EC%B4%88.ipynb)
22. [합성곱신경망 특성맵시각화](22_%ED%95%A9%EC%84%B1%EA%B3%B1%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%8A%B9%EC%84%B1%EB%A7%B5%EC%8B%9C%EA%B0%81%ED%99%94.ipynb)
23. [합성곱신경망 MNIST분류](23_%ED%95%A9%EC%84%B1%EA%B3%B1%EC%8B%A0%EA%B2%BD%EB%A7%9D_MNIST%EB%B6%84%EB%A5%98.ipynb)
24. [강화학습 개요](24_%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5_%EA%B0%9C%EC%9A%94.ipynb)
25. [합성곱신경망 CIFAR10분류](25_%ED%95%A9%EC%84%B1%EA%B3%B1%EC%8B%A0%EA%B2%BD%EB%A7%9D_CIFAR10%EB%B6%84%EB%A5%98.ipynb)
26. [순환신경망 구조와은닉상태](26_%EC%88%9C%ED%99%98%EC%8B%A0%EA%B2%BD%EB%A7%9D_%EA%B5%AC%EC%A1%B0%EC%99%80%EC%9D%80%EB%8B%89%EC%83%81%ED%83%9C.ipynb)
27. [순환신경망 IMDB임베딩분류](27_%EC%88%9C%ED%99%98%EC%8B%A0%EA%B2%BD%EB%A7%9D_IMDB%EC%9E%84%EB%B2%A0%EB%94%A9%EB%B6%84%EB%A5%98.ipynb)
28. [순환신경망 LSTM과GRU비교](28_%EC%88%9C%ED%99%98%EC%8B%A0%EA%B2%BD%EB%A7%9D_LSTM%EA%B3%BCGRU%EB%B9%84%EA%B5%90.ipynb)
29. [로이터뉴스 카테고리분류](29_%EB%A1%9C%EC%9D%B4%ED%84%B0%EB%89%B4%EC%8A%A4_%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC%EB%B6%84%EB%A5%98.ipynb)
30. [텍스트분류 CNN과RNN결합](30_%ED%85%8D%EC%8A%A4%ED%8A%B8%EB%B6%84%EB%A5%98_CNN%EA%B3%BCRNN%EA%B2%B0%ED%95%A9.ipynb)
31. [네이버영화리뷰 감성분석](31_%EB%84%A4%EC%9D%B4%EB%B2%84%EC%98%81%ED%99%94%EB%A6%AC%EB%B7%B0_%EA%B0%90%EC%84%B1%EB%B6%84%EC%84%9D.ipynb)

### PyTorch·전이학습과 자연어 처리

32. [파이토치 텐서와데이터로더](32_%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98_%ED%85%90%EC%84%9C%EC%99%80%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A1%9C%EB%8D%94.ipynb)
33. [파이토치 자동차평가분류](33_%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98_%EC%9E%90%EB%8F%99%EC%B0%A8%ED%8F%89%EA%B0%80%EB%B6%84%EB%A5%98.ipynb)
34. [파이토치 패션이미지분류](34_%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98_%ED%8C%A8%EC%85%98%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98.ipynb)
35. [이미지분류 신경망구조비교](35_%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98_%EC%8B%A0%EA%B2%BD%EB%A7%9D%EA%B5%AC%EC%A1%B0%EB%B9%84%EA%B5%90.ipynb)
36. [전이학습 파이토치개고양이분류](36_%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5_%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98%EA%B0%9C%EA%B3%A0%EC%96%91%EC%9D%B4%EB%B6%84%EB%A5%98.ipynb)
37. [전이학습 케라스개고양이분류](37_%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5_%EC%BC%80%EB%9D%BC%EC%8A%A4%EA%B0%9C%EA%B3%A0%EC%96%91%EC%9D%B4%EB%B6%84%EB%A5%98.ipynb)
38. [토크나이징 형태소분석](38_%ED%86%A0%ED%81%AC%EB%82%98%EC%9D%B4%EC%A7%95_%ED%98%95%ED%83%9C%EC%86%8C%EB%B6%84%EC%84%9D.ipynb)
39. [임베딩 단어벡터기초](39_%EC%9E%84%EB%B2%A0%EB%94%A9_%EB%8B%A8%EC%96%B4%EB%B2%A1%ED%84%B0%EA%B8%B0%EC%B4%88.ipynb)
40. [단어임베딩 Word2Vec학습](40_%EB%8B%A8%EC%96%B4%EC%9E%84%EB%B2%A0%EB%94%A9_Word2Vec%ED%95%99%EC%8A%B5.ipynb)
41. [텍스트 유사도](41_%ED%85%8D%EC%8A%A4%ED%8A%B8_%EC%9C%A0%EC%82%AC%EB%8F%84.ipynb)
42. [문답데이터 감정분류](42_%EB%AC%B8%EB%8B%B5%EB%8D%B0%EC%9D%B4%ED%84%B0_%EA%B0%90%EC%A0%95%EB%B6%84%EB%A5%98.ipynb)
43. [개체명인식 양방향LSTM](43_%EA%B0%9C%EC%B2%B4%EB%AA%85%EC%9D%B8%EC%8B%9D_%EC%96%91%EB%B0%A9%ED%96%A5LSTM.ipynb)

### LLM·Transformer·BERT, 모델 해석과 보조 실습

44. [대규모언어모델 개념](44_%EB%8C%80%EA%B7%9C%EB%AA%A8%EC%96%B8%EC%96%B4%EB%AA%A8%EB%8D%B8_%EA%B0%9C%EB%85%90.ipynb)
45. [트랜스포머 토큰화](45_%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8_%ED%86%A0%ED%81%B0%ED%99%94.ipynb)
46. [트랜스포머 사전학습모델](46_%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8_%EC%82%AC%EC%A0%84%ED%95%99%EC%8A%B5%EB%AA%A8%EB%8D%B8.ipynb)
47. [버트 문서분류](47_%EB%B2%84%ED%8A%B8_%EB%AC%B8%EC%84%9C%EB%B6%84%EB%A5%98.ipynb)
48. [설명가능한인공지능 모델해석](48_%EC%84%A4%EB%AA%85%EA%B0%80%EB%8A%A5%ED%95%9C%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5_%EB%AA%A8%EB%8D%B8%ED%95%B4%EC%84%9D.ipynb)
49. [삼각함수 사인과코사인](49_%EC%82%BC%EA%B0%81%ED%95%A8%EC%88%98_%EC%82%AC%EC%9D%B8%EA%B3%BC%EC%BD%94%EC%82%AC%EC%9D%B8.ipynb)
