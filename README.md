# UCI-SECOM-
UCI SECOM 공정 데이터 기반 불량 예측 및 관리 우선순위 변수 탐색 프로젝트를 진행하는 저장소입니다
# UCI SECOM 공정 데이터 기반 불량 예측 및 관리 우선순위 변수 탐색

## 1. Project Overview

본 프로젝트는 **UCI SECOM 반도체 공정 데이터**를 활용하여 공정 측정 데이터를 탐색하고, 불량 예측 모델을 구축한 뒤, **불량 예측에 높은 기여도를 보이는 측정 변수 후보를 선별하여 관리 우선순위를 제안하는 것**을 목표로 한다.

단순한 머신러닝 모델 정확도 경쟁이 아니라, 실제 반도체 **양산기술 / 공정기술 엔지니어의 데이터 기반 문제 해결 관점**에서 다음 역량을 보여주는 것을 목표로 한다.

* 공정 데이터 탐색 및 구조 파악
* 데이터 품질 검증
* Spotfire 기반 시각화 및 EDA
* Python 기반 데이터 전처리
* 결측치 처리 전략 비교
* 불균형 데이터 대응
* 머신러닝 기반 불량 예측
* 중요 변수 후보 선별
* 분석 결과의 공정 엔지니어 관점 해석

---

## 2. Research Question

> **수백 개의 익명화된 반도체 공정 측정 변수 가운데 불량 예측에 유효한 변수를 어떻게 선별하고, 엔지니어가 우선적으로 살펴볼 관리 후보로 압축할 수 있는가?**

SECOM 데이터의 측정 변수는 익명화되어 있기 때문에 특정 변수의 실제 물리적 의미나 공정 원인을 확인할 수 없다.

따라서 본 프로젝트에서는 Feature Importance 또는 통계적 유의성을 근거로 특정 변수를 **“불량의 원인”**이라고 해석하지 않는다.

최종 분석 결과는 다음과 같이 제한하여 해석한다.

> **불량 예측에 높은 기여도를 보인 변수 후보를 선별하고, 추가적인 공정 검토가 필요한 관리 우선순위를 제안한다.**

---

## 3. Dataset

### UCI SECOM Dataset

반도체 제조 공정에서 수집된 다수의 공정 측정 변수와 최종 Pass / Fail 정보를 포함하는 데이터셋이다.

주요 특징:

* 약 1,500개의 생산 샘플
* 수백 개의 익명화된 공정 측정 변수
* Pass / Fail Label
* 다수의 Missing Value
* Pass 대비 Fail 데이터가 적은 Class Imbalance
* 시간 정보 포함

변수의 실제 센서명과 공정 단계는 공개되어 있지 않기 때문에 본 프로젝트에서는 데이터에서 직접 확인 가능한 특성만을 기반으로 분석한다.

---

# 4. Project Principles

## 4.1 Raw Data Preservation

원본 데이터는 절대로 수정하지 않는다.

```text
uci-secom.csv
```

원본 파일을 기준으로 모든 분석을 다시 수행하며, 전처리된 데이터는 별도의 Output 파일로 저장한다.

초기 분석 과정에서 생성했던 다음 데이터는 시행착오 기록으로만 보존한다.

```text
0_fill
median_cleaned
```

최종 분석에는 사용하지 않는다.

---

## 4.2 EDA와 Modeling 전처리 분리

### EDA

탐색적 데이터 분석에서는 원본 상태를 최대한 보존한다.

* NaN 유지
* 실제 0 유지
* Time 유지
* Pass / Fail 유지

Python과 Spotfire를 이용하여 데이터 자체의 상태와 패턴을 먼저 확인한다.

### Modeling

머신러닝에서는 Raw Data에서 다시 출발한다.

```text
Raw Data
   ↓
X / y 분리
   ↓
Train / Test Split
   ↓
Train 기준 전처리 학습
   ↓
Test에 동일한 규칙 적용
   ↓
Model Training
```

전처리 과정에서 Test 데이터의 정보를 미리 사용하는 **Data Leakage**를 방지하는 것을 핵심 원칙으로 한다.

---

## 4.3 Missing Value

프로젝트 초기에는 모든 NaN 값을 다음과 같이 처리하였다.

```python
fillna(0)
```

그러나 Spotfire 분석 과정에서 0 부근에 인위적으로 값이 집중되는 현상과 극단적인 통계 결과가 다수 관찰되었다.

이에 따라 다음과 같이 접근 방식을 수정하였다.

> **0 대치 이후 극단적인 통계 결과가 다수 관찰되어 전처리 방식의 영향을 의심했고, Raw Data부터 다시 검토하기로 했다.**

0 대치가 특정 통계 결과의 직접적인 원인이라고 단정하지 않는다.

또한 익명화된 SECOM 데이터에서는 Missing Value가 발생한 실제 이유를 확인할 수 없으므로,

* 센서 고장
* 미사용 Channel
* MCAR
* MAR
* MNAR

등의 결측 원인을 임의로 단정하지 않는다.

대신 데이터에서 직접 확인 가능한 특성만 사용한다.

예:

```text
100% Missing → All-Missing Feature
값이 한 종류 → Constant Feature
```

---

## 4.4 Missing Value Strategy

Median Imputation을 실제 값을 복원하는 정답으로 해석하지 않는다.

Median은 이상치의 영향을 상대적으로 덜 받는 **단순 Baseline Imputation**으로 사용한다.

두 가지 전략을 비교할 예정이다.

### Strategy A

```text
Raw NaN
→ XGBoost Native Missing Handling
```

### Strategy B

```text
Median Imputation
+
Missing Indicator
→ XGBoost
```

Missing Indicator는 결측값을 Median으로 대체하면서도,

> “이 값은 원래 Missing이었다.”

라는 정보를 별도의 0 / 1 변수로 보존하는 방법이다.

KNN Imputation, MissForest, Multiple Imputation 등의 방법은 필수 분석으로 사용하지 않고 필요 시 추가 실험으로 고려한다.

---

## 4.5 Imbalanced Data

SECOM 데이터는 Pass 데이터가 Fail 데이터보다 훨씬 많은 불균형 데이터이다.

따라서 모델 평가에서 Accuracy만 사용하지 않는다.

주요 평가 지표:

* Confusion Matrix
* Recall
* Precision
* F1-score
* PR-AUC

불균형 대응은 다음 순서로 비교한다.

```text
1. Baseline XGBoost
2. Class Weight / scale_pos_weight
3. 필요 시 SMOTE
```

SMOTE를 적용할 경우 반드시 Train 영역 내부에서만 수행하여 Data Leakage를 방지한다.

---

## 4.6 Feature Importance ≠ Causality

머신러닝에서 높은 Feature Importance를 가진 변수는

```text
불량을 발생시킨 원인
```

으로 해석하지 않는다.

본 프로젝트에서는 다음과 같이 표현한다.

```text
모델의 불량 예측에 높은 기여도를 보인 변수 후보
```

이후 Spotfire를 이용하여 해당 변수의

* Distribution
* Pass / Fail 차이
* Time Trend
* Missing Pattern

등을 다시 확인한다.

---

# 5. Project Workflow

## Phase 0 — Project Structure

프로젝트 파일을 역할에 따라 분리하고 Raw Data를 보호한다.

```text
SECOM_Project/
│
├─ Raw/
│
├─ Notebook/
│
├─ Output/
│
├─ Spotfire/
│
├─ Figures/
│
├─ Portfolio/
│
└─ README.md
```

---

## Phase 1 — Raw Data Audit

Raw Data의 기본 구조와 데이터 품질을 확인한다.

확인 항목:

* Dataset Shape
* Sensor Feature 개수
* Pass / Fail 개수
* Pass / Fail 비율
* 전체 Missing Value 개수
* 전체 Missing Rate
* Sensor별 Missing Rate
* All-Missing Feature
* Constant Feature
* Raw Data 내 실제 숫자 0 존재 여부

이 단계에서는 NaN을 대체하지 않는다.

---

## Phase 2 — Spotfire EDA

Raw-preserved 데이터를 Spotfire에서 분석한다.

주요 분석:

### 1. Pass / Fail Ratio

불량 데이터의 비율과 Class Imbalance 확인

### 2. Sensor Missing Rate

센서별 Missing Value 분포 확인

### 3. Time vs Sensor Trend

시간에 따른 Sensor 값 변화를 확인하고 Pass / Fail을 색상으로 구분한다.

### 4. Pass / Fail Box Plot

두 그룹 간 Sensor 분포 차이를 탐색한다.

### 5. Data Relationships / ANOVA

통계적으로 Pass / Fail 그룹 간 차이가 나타나는 변수 후보를 탐색한다.

ANOVA 결과는 불량 원인을 확인하는 용도가 아니라 **Feature Screening** 목적으로만 사용한다.

---

## Phase 3 — Leakage-Free ML Pipeline

Raw Data에서 다시 시작한다.

```text
Raw Data
↓
Feature / Target 분리
↓
Train / Test Split
↓
Train 기준 Feature Cleaning
↓
Train 기준 Preprocessing
↓
Test Transform
↓
Model Training
↓
Evaluation
```

Train 데이터를 기준으로

* All-Missing Feature 제거
* Constant Feature 제거
* Imputation
* Missing Indicator 생성

등을 수행한다.

초기 Baseline에서는 Time을 예측 변수에서 제외하고 EDA 용도로 활용한다.

---

## Phase 4 — Missing Strategy + XGBoost

두 가지 Missing Value 처리 전략을 비교한다.

```text
A. XGBoost Native Missing Handling

B. Median Imputation
   + Missing Indicator
   + XGBoost
```

이후 `scale_pos_weight`를 적용하여 불균형 대응 효과를 비교한다.

---

## Phase 5 — Model Evaluation

모델 성능은 Accuracy 중심으로 평가하지 않는다.

```text
Confusion Matrix
Recall
Precision
F1-score
PR-AUC
```

특히 제조 공정 관점에서

```text
Fail을 놓치는 비용
vs
정상 제품을 Fail로 오판하는 비용
```

사이의 Trade-off를 분석한다.

---

## Phase 6 — Important Feature Candidates

최종 모델에서 불량 예측에 높은 기여도를 보이는 변수 후보를 선별한다.

또한 다음 두 분석 결과를 비교한다.

```text
Spotfire ANOVA / Data Relationships
vs
XGBoost Feature Importance
```

단변량 통계 분석과 다변량 머신러닝 모델에서

* 공통적으로 중요하게 나타나는 변수
* 한 분석에서만 중요하게 나타나는 변수

를 비교한다.

Top N은 분석 이전에 고정하지 않는다.

최종 Dashboard에서는 가독성을 고려하여 약 10~20개의 주요 관리 후보를 표현하는 방향을 검토한다.

---

## Phase 7 — Spotfire Re-validation

Python에서 선별된 주요 Feature를 Spotfire에서 다시 확인한다.

주요 검토 항목:

```text
Time Trend
Pass / Fail Box Plot
Distribution
Missing Pattern
```

이를 기반으로 최종적으로

> **추가적인 공정 검토가 필요한 관리 우선순위 변수 후보**

를 제안한다.

---

## Phase 8 — Portfolio

최종 결과물을 취업 포트폴리오 형태로 정리한다.

### Deliverables

* Jupyter Notebook
* Spotfire Dashboard
* Model 성능 비교표
* Feature Importance 결과
* GitHub Repository
* Blog Series
* 취업용 Portfolio
* 면접 예상 질문 및 답변

### Blog Series

1. SECOM 데이터와 잘못된 0 대치
2. Spotfire 기반 Raw Data EDA
3. Data Leakage를 방지한 ML Pipeline
4. XGBoost 기반 반도체 불량 예측
5. 중요 변수 후보 분석 및 Spotfire Dashboard

---

# 6. Tools

### Data Analysis

```text
Python
Pandas
NumPy
Jupyter Notebook
```

### Machine Learning

```text
scikit-learn
XGBoost
```

### Visualization / EDA

```text
TIBCO Spotfire
Matplotlib
```

---

# 7. Current Progress

현재 프로젝트 진행 단계:

```text
Phase 0  Project Structure      ✅
Phase 1  Raw Data Audit         🚧
Phase 2  Spotfire EDA
Phase 3  ML Pipeline
Phase 4  Missing Strategy
Phase 5  Model Evaluation
Phase 6  Feature Candidates
Phase 7  Spotfire Re-validation
Phase 8  Portfolio
```

현재는 기존 전처리 데이터가 아닌 원본 `uci-secom.csv`에서 다시 시작하여 **Raw Data Audit**을 수행하고 있다.

---

# 8. Expected Outcome

본 프로젝트의 최종 목표는 최고의 머신러닝 정확도를 만드는 것이 아니다.

반도체 공정 데이터를 대상으로

```text
Data Audit
↓
EDA
↓
Data Quality Verification
↓
Leakage-Free Preprocessing
↓
Failure Prediction
↓
Important Feature Screening
↓
Spotfire Re-validation
↓
Management Priority Proposal
```

의 전체 분석 과정을 직접 수행함으로써,

> **공정 데이터를 기반으로 문제를 정의하고, 데이터를 검증하며, 분석 결과를 엔지니어링 관점의 관리 후보로 연결하는 능력**

을 보여주는 것을 목표로 한다.

---

## Keywords

`Semiconductor` `SECOM` `Manufacturing Data` `Process Engineering` `Yield` `Defect Prediction` `Spotfire` `Python` `XGBoost` `Missing Data` `Imbalanced Data` `Feature Importance`
