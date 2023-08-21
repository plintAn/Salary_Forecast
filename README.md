# Salary_Forecast

이번 시간에는 미국 인종 별 월급 예측을 위한 분석을 진행해보겠습니다.

# DataSet(National Longitudinal Survey of Youth)
1997-2011 년 청소년 전국 종단 조사 (National Longitudinal Survey of Youth 1997-2011) 데이터 세트는 미국 데이터로 작업하는 사회 과학자들이 이용할 수있는 가장 중요한 데이터베이스 중 하나입니다.
이를 통해 과학자들은 소득과 교육 성취의 결정 요인을 볼 수 있으며 정부 정책과 놀라운 관련성을 가지고 있습니다. 또한 인종, 성별 및 기타 요인이 다른 사람들의 교육 수준과 급여가 얼마나 다른지와 같은 정치적으로 민감한 문제를 밝힐 수 있습니다. 이러한 변수가 교육과 소득에 어떤 영향을 미치는지 더 잘 이해하면 더 적합한 정부 정책을 수립 할 수 있습니다.


# 목차

* 데이터 로드
* 데이터 전처리(정제 및 가공)
* 독립변수와 종속변수를 선택
* 로지스틱 회귀 모델을 생성
* 로지스틱 회귀 모델을 평가

## 데이터 로드 전 사전 작업

* 라이브러리 임포트
```python
import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```
단위 수정(소수점 2번쨰자리까지 표시, 천단위 ','표시)
```python
pd.options.display.float_format = '{:,.2f}'.format
```


# 데이터 로드


```python
df = pd.read_csv('Dataset.csv')
```

# 데이터 전처리

## 데이터 Nan 값, 중복값, 

데이터 확인

```python
df.describe()
df.shape

8 rows × 96 columns
(2000, 96)
```

* 데이터 크기가 매우 작은 편이니, 결측 값은 최대한 유지하는 전처리(평균, 중앙값, 최빈값으로 채우기) 방식으로 진행
* ID의 경우 확인 결과 그냥 고유 번호이니 제외시킵니다.
* 2000rows ==> 1487 rows로 513rows의 많은 중복 값이 있어 확인 결과 중복값이 맞음.

```python
# 'ID' 컬럼 삭제
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# 중복된 행 제거
df = df.drop_duplicates(subset=['ASVABMK', 'ASVABC'])

# 연속형 변수의 경우 평균으로 결측값을 대체
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].mean(), inplace=True)
    


# 범주형 변수의 경우 최빈값으로 결측값을 대체
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)


# 데이터 저장
df.to_csv('data_processed.csv', index=False)

```


# 독립변수와 종속변수를 선택

* 95 row x 1487 columns에 대한 분석은 어려울 것으로 판단하여, 독립변수(Earnings)와 96개 종속변수의 중요도를 계산
* 상위 10개 종속 변수 확인

```python
# 독립변수와 종속변수의 이름
independent_variable = 'EARNINGS'
dependent_variables = df.columns.difference([independent_variable])

# 변수의 중요도를 측정
features = df[dependent_variables]
importances = features.apply(lambda x: x.var())

# 중요도를 기준으로 변수를 정렬
importances = importances.sort_values(ascending=False)

# 상위 10개 변수만 선택
top_10_dependent_variables = importances.head(10).index

# 상관관계를 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for col in top_10_dependent_variables:
    plt.scatter(df[independent_variable], df[col], label=col)
    
plt.xlabel(independent_variable)
plt.ylabel('Top 10 Dependent Variables')
plt.legend()
plt.show()

```
![image](https://github.com/plintAn/Salary_Forecast/assets/124107186/279e6f86-7ce8-46f5-808c-c8dca724ed86)

* 분산을 이용하 'Earnings'에 중요도 상위 10개 확인

```python
# 주어진 변수 리스트
top_10_dependent_variables = ['HHINC97', 'ASVABMV', 'POVRAT97', 'WEIGHT11', 'WEIGHT04', 'HOURS',
       'AGEMBTH', 'HEIGHT', 'JOBS', 'SF']

# 주어진 변수들의 중요도 (분산) 계산
selected_importances = features[top_10_dependent_variables].apply(lambda x: x.var())

# 중요도를 내림차순으로 정렬
sorted_importances = selected_importances.sort_values(ascending=False)

print(sorted_importances)
```

OutPut

```python
HHINC97    1,489,001,150.26
ASVABMV      779,563,410.46
POVRAT97          66,516.57
WEIGHT11           2,169.67
WEIGHT04           1,709.02
HOURS                121.47
AGEMBTH               24.83
HEIGHT                16.89
JOBS                  10.53
SF                     8.88
dtype: float64
```

* 분산 확인 후 상위 5개 종속 변수 활용하기로 결정.


```python
top_5_dependent_variables
```
# 로지스틱 회귀 모델을 생성



```python
top_5_dependent_variables = ['HHINC97', 'ASVABMV', 'POVRAT97', 'WEIGHT11', 'WEIGHT04']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 독립변수 설정: 상위 5개 변수를 모두 사용
X = df[top_5_dependent_variables]

# 각 종속 변수에 대한 모델 훈련 및 평가
for dependent_variable in top_5_dependent_variables:
    # 중앙값을 기준으로 0과 1의 두 범주로 변환
    median_value = df[dependent_variable].median()
    y = (df[dependent_variable] > median_value).astype(int)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 로지스틱 회귀 모델 생성 및 훈련
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 모델 평가
    accuracy = model.score(X_test, y_test)
    
    print(f"Dependent Variable: {dependent_variable}")
    print(f'Accuracy: {accuracy:.2f}')
    print('-'*50)


```
OutPut
```python
Dependent Variable: HHINC97
Accuracy: 0.84
--------------------------------------------------
Dependent Variable: ASVABMV
Accuracy: 0.90
--------------------------------------------------
Dependent Variable: POVRAT97
Accuracy: 0.87
--------------------------------------------------
Dependent Variable: WEIGHT11
Accuracy: 0.69
--------------------------------------------------
Dependent Variable: WEIGHT04
Accuracy: 0.73
--------------------------------------------------
```


## 다른 평가 지표(정밀도, 재현율) 시각화(ROC 곡선, 혼동 행렬) 진행

* 정밀도(Precision): Positive로 예측한 것 중 실제 Positive의 비율.
* 재현율(Recall): 실제 Positive 중 Positive로 예측한 비율.
* ROC 곡선: 분류 임계값에 따른 재현율과 특이도를 그래프로 표현. 좋은 모델은 왼쪽 상단에 가까움.
* 혼동 행렬(Confusion Matrix): 예측값과 실제값의 일치 여부를 표로 나타냄.

간단히 말하면, 정밀도와 재현율은 모델이 얼마나 잘 예측했는지를 수치로 나타내며, ROC 곡선은 여러 임계값에서의 성능을 보여주고, 혼동 행렬은 예측의 정확성을 표로 제공한다

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, confusion_matrix

# 독립변수 설정: 상위 5개 변수를 모두 사용
X = df[top_5_dependent_variables]

# 각 종속 변수에 대한 모델 훈련 및 평가
for dependent_variable in top_5_dependent_variables:
    # 중앙값을 기준으로 0과 1의 두 범주로 변환
    median_value = df[dependent_variable].median()
    y = (df[dependent_variable] > median_value).astype(int)  # 변수의 중앙값을 기준으로 0과 1로 데이터 변환
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  # 75%는 훈련 데이터, 25%는 테스트 데이터로 분리
    
    # 로지스틱 회귀 모델 생성 및 훈련
    model = LogisticRegression()  # 로지스틱 회귀 모델 인스턴스 생성
    model.fit(X_train, y_train)  # 훈련 데이터로 모델 훈련
    
    # 모델 평가
    accuracy = model.score(X_test, y_test)  # 정확도 계산
    precision = precision_score(y_test, model.predict(X_test))  # 정밀도 계산
    recall = recall_score(y_test, model.predict(X_test))  # 재현율 계산
    f1 = f1_score(y_test, model.predict(X_test))  # F1 스코어 계산
    
    print(f"Dependent Variable: {dependent_variable}")
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1: {f1:.2f}')
    
    # ROC 곡선 그리기
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])  # ROC 곡선을 위한 값 계산
    plt.plot(fpr, tpr)  # ROC 곡선 표시
    plt.xlabel('FPR')  # x축 레이블
    plt.ylabel('TPR')  # y축 레이블
    plt.title('ROC Curve')  # 그래프 제목
    plt.show()
    
    # 혼동 행렬 그리기
    cm = confusion_matrix(y_test, model.predict(X_test))  # 혼동 행렬 계산
    plt.matshow(cm)  # 혼동 행렬 그리기
    plt.xlabel('True Label')  # x축 레이블
    plt.ylabel('Predicted Label')  # y축 레이블
    plt.title('Confusion Matrix')  # 그래프 제목
    plt.show()

```

### 코드 설명

* 각 변수에 대하여 로지스틱 회귀를 사용하여 모델링을 진행
* 각 변수의 중앙값을 기준으로 이진 분류 문제로 변환
* 데이터를 훈련 데이터와 테스트 데이터로 분리합니다.
* 훈련 데이터로 로지스틱 회귀 모델을 학습시키고, 테스트 데이터로 평가
* 각 종속 변수에 대하여 정확도, 정밀도, 재현율, F1 스코어를 계산하고 출력
* 로지스틱 회귀 분석과의 비교를 위해 ROC 곡선을 그려 성능을 시각화
* 예측 결과를 혼동 행렬로도 시각화하여 직관적으로 확인(여기서 중요한 것은 각 변수를 중앙값을 기준으로 이진 분류 문제로 변환한 후, 해당 변수를 예측하는 모델을 학습하는 방식을 사용했습니다)

OutPut :

```python
Dependent Variable: HHINC97
Accuracy: 0.84
Precision: 0.78
Recall: 0.69
F1: 0.73
```
![image](https://github.com/plintAn/Salary_Forecast/assets/124107186/ab1e28ef-11ef-421d-9649-b3ab668d1fb1)

OutPut :

```python
Dependent Variable: ASVABMV
Accuracy: 0.90
Precision: 0.88
Recall: 0.92
F1: 0.90
```
![image](https://github.com/plintAn/Salary_Forecast/assets/124107186/75c9ccb7-80f6-411c-bbde-cc91c154489d)

OutPut :

```python
Dependent Variable: ASVABMV
Accuracy: 0.90
Precision: 0.88
Recall: 0.92
F1: 0.90
```

![image](https://github.com/plintAn/Salary_Forecast/assets/124107186/25ab04fb-4bbf-4a19-b54c-b8df11eaf4ab)


OutPut :

```python
Dependent Variable: ASVABMV
Accuracy: 0.90
Precision: 0.88
Recall: 0.92
F1: 0.90
```

![image](https://github.com/plintAn/Salary_Forecast/assets/124107186/de0fb66b-e56f-46ea-9581-3c98bed89c9c)

OutPut :

```python
Dependent Variable: WEIGHT04
Accuracy: 0.73
Precision: 0.71
Recall: 0.78
F1: 0.74
```

![image](https://github.com/plintAn/Salary_Forecast/assets/124107186/f2edac2d-8d17-4536-9f83-19d4e8118cab)

## 혼동 행렬 해석

위의 혼동 행렬과 평가 지표를 종합해보면, ASVABMV의 성능이 가장 좋고, WEIGHT11의 성능이 좋지 않다. ASVABMV의 경우 Accuracy, Precision, Recall, F1 모두 높게 나타났는데, 이는 ASVABMV에 대한 모델이 잘 학습되었고, 실제값과 잘 예측하고 있다는 것이다. WEIGHT11의 경우 Accuracy는 높지만, Precision과 Recall이 낮게 나타났는데, 이는 모델이 실제값을 양성으로 예측하는 비율은 높지만, 실제 양성 중 양성으로 예측하는 비율은 낮음을 의미한다.




# 로지스틱 회귀 모델을 평가


데이터의 결과

HHINC97, ASVABMV, POVRAT97: 이 세 변수는 모두 높은 정확도를 가지므로, 주어진 독립 변수를 예측이 성공적에 특히 ASVABMV는 90%의 매우 높은 정확도를 가지므로, 가장 강한 연관성을 가진 것으로 볼 수 있다.

WEIGHT11, WEIGHT04: 이 두 변수는 70% 미만의 상대적으로 낮은 정확도를 보이는데, 이는 데이터 수가 부족해서 나타난 상황으로 해석된다.

종합: ASVABMV와 POVRAT97 변수는 다른 변수들과 강한 연관성이 있어 잘 예측하고 있다. 반면, WEIGHT11과 WEIGHT04는 데이터 부족으로 인해 어렵다.


## 학사 학위(12+4)년의 교육과 5년의 경력을 가진 사람 수익 예측

```python
# Feature selection
X = df[["S", "EXP"]]
y = df["EARNINGS"]

# Model training
model = LinearRegression()
model.fit(X, y)

# Prediction
pred = model.predict(np.array([[12, 5]]))

# Print the result
print(pred)

```

이 결과를 보면, 학사 학위(12+4)년의 교육과 5년의 경력을 가진 사람이 2011년에 벌 수 있는 예상 급여는 $50,000으로 예측할 수 있다.














