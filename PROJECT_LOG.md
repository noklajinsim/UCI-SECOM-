# PROJECT LOG — UCI SECOM 공정 불량 예측 및 관리 우선순위 변수 탐색

> 이 문서는 GitHub 첫 화면용 요약이 아니라, 프로젝트 수행 과정의 **판단 근거·실패·수정·의사결정 흐름을 보존하기 위한 상세 기록**이다.  
> README에는 결과와 핵심 메시지만 압축하고, 여기서는 **“왜 다음 실험을 했는가?”**가 끊기지 않도록 남긴다.

---

# 0. 프로젝트의 출발점

UCI SECOM 데이터에는 1,567개의 Sample과 590개의 익명화된 반도체 공정 Sensor가 존재한다.

처음에는 단순히 다음 문제로 접근할 수도 있었다.

> “590개의 Sensor로 Pass / Fail을 얼마나 잘 예측할 수 있는가?”

그러나 이 프로젝트의 목적은 데이터 사이언스 경진대회가 아니라 **반도체 양산·공정기술 취업용 포트폴리오**였기 때문에 질문을 확장했다.

> **“590개의 익명 공정 변수 가운데 불량 예측에 유효한 후보를 어떻게 선별하고, 엔지니어가 우선적으로 확인할 관리 후보로 압축할 수 있는가?”**

따라서 목표를 두 층으로 나눴다.

1. **불량 예측 모델을 구축하고 일반화 성능을 검증한다.**
2. **통계·ML·CV 안정성·Spotfire 재검증을 결합해 관리 우선순위 후보를 도출한다.**

가장 중요한 해석 원칙은 다음과 같다.

> **Feature Importance ≠ Failure Cause**

Sensor 의미가 익명화되어 있고 Equipment / Recipe / Lot / Process Step Metadata가 없으므로, 중요 변수를 “불량 원인”이라고 단정하지 않는다.

---

# 1. 분석 전 고정한 원칙

1. **Raw Data는 수정하지 않는다.**  
   `01_raw/uci-secom.csv`를 원본 그대로 보존한다.

2. **Missing과 실제 0을 구분한다.**  
   NaN은 “측정값 없음”, 0은 실제 관측값일 수 있다.

3. **EDA와 Modeling 전처리를 분리한다.**  
   EDA에서는 NaN과 실제 값을 최대한 그대로 유지한다.

4. **Train / Test Split 이후 모델링 의사결정을 한다.**  
   Imputation, Oversampling, Feature Selection, Hyperparameter Tuning, Threshold Selection은 Train 또는 Train 내부 CV에서만 수행한다.

5. **Validation / Test에는 Oversampling하지 않는다.**

6. **Accuracy를 주요 지표로 사용하지 않는다.**  
   Fail 비율이 약 6.6%이므로 모두 Pass로 예측해도 약 93% Accuracy가 나온다.

7. **모델 선택과 Threshold 선택을 분리한다.**  
   - 모델 ranking 비교 → Average Precision(AP)  
   - 운영점 결정 → Precision / Recall / F1 trade-off

8. **최종 Test 결과를 확인한 뒤 모델을 다시 조정하지 않는다.**

---

# 2. 전체 의사결정 지도

| 관찰 / 문제 | 위험 | 수행한 실험 | 결과 | 결정 |
|---|---|---|---|---|
| Missing 존재 | Missing을 실제 0과 혼동 | Raw audit | 전체 결측률 4.52%, 일부 Feature에 집중 | Missing과 0 분리 |
| Missing을 0으로 대치 | 평균/ANOVA 왜곡 | 원본과 재비교 | 비정상적으로 작은 p-value 다수 | 0 대치 폐기, Raw 복원 |
| Fail 6.6% | Accuracy 착시 | Baseline XGB | Test Fail 21개 모두 놓침 | AP/Recall/F1 중심 평가 |
| Native NaN만 사용 | Missing pattern 정보 손실 가능 | Median+Indicator 비교 | B계열 상대적으로 유망 | Median+Indicator 채택 |
| Class Weight | Fail 탐지 개선 제한 | `scale_pos_weight` | Recall 개선 미미 | Oversampling 검토 |
| Missing Indicator가 binary | 일반 SMOTE가 0~1 사이 가상값 생성 | SMOTENC | Indicator를 categorical 처리 | SMOTENC 채택 |
| Sensor scale 차이 | 거리 기반 이웃 왜곡 | RobustScaler | 이웃 탐색 공간 보정 | Sensor에만 scaling |
| B2 성능 낮음 | 설정 미조정 | XGB/SMOTENC tuning | OOF AP 0.1769 → 0.2050 | 최종 모델 후보 확정 |
| 590 Feature | 잡음 Feature 가능 | Train-only Top-K ANOVA | 590 전체가 최고 | Feature Selection 미채택 |
| OOF 개선 | CV 결과 과신 위험 | Independent Test | AP 0.2050 → 0.1213 | 생산 적용 모델 주장 철회 |
| 낮은 Test 성능 | Importance 과신 위험 | Fold-wise importance | 반복 상위 Feature 확인 | 안정 후보만 해석 |
| 통계와 ML 결과 불일치 | 단일 방법론 의존 | ANOVA × XGB 비교 | Multi-evidence / ML-only 유형 | Evidence type별 후보 분류 |
| Sensor 129 유의 | 시간 효과 혼재 가능 | Spotfire time validation | 9월 Pass 급락 | 인과 단정 금지 |

---

# 3. Phase 1 — Raw Data Audit

## 3.1 데이터 구조

- Samples: **1,567**
- Columns: **592**
  - `Time`: 1
  - Sensor: **590**
  - `Pass/Fail`: 1
- Pass (`-1`): **1,463 (93.36%)**
- Fail (`1`): **104 (6.64%)**

처음부터 강한 클래스 불균형이 존재했다.

## 3.2 Missing Audit

- Total Missing: **41,951**
- Total Cells: **927,664**
- Overall Missing Rate: **4.52%**

전체 4.52%만 보면 문제가 작아 보이지만, Feature별 결측률은 매우 불균일했다.

Train Split 이후에도:

- Missing ≥ 50%: 24개
- Missing ≥ 80%: 8개
- Missing ≥ 90%: 4개

가 존재했다.

따라서 “전체 결측률이 낮다 → 결측 문제는 중요하지 않다”라고 결론 내리지 않았다.

## 3.3 Constant / Zero Audit

Raw 기준:

- All-Missing Feature: **0개**
- Strict Constant Feature: **0개**
- Observed-value Constant + Missing: **116개**
- 실제 Zero 값: **200,005개**
- Zero를 포함하는 Feature: **215개**

여기서 중요한 구분이 생겼다.

어떤 Feature가 관측된 값은 모두 100이고 일부가 NaN이라면, 이는 Strict Constant가 아니라 **Observed-value Constant + Missing**이다.

Missing 자체가 정보일 가능성이 있으므로 이 Feature들을 초기부터 무조건 제거하지 않았다.

---

# 4. 첫 번째 실패 — Missing을 0으로 대체

## 4.1 처음의 접근

초기에는 분석 편의를 위해 Missing을 모두 `0`으로 대체했다.

그러나 실제 데이터에는 이미 **200,005개의 실제 0값**이 존재했다.

따라서 다음 두 의미가 섞였다.

```text
0 = 실제 Sensor 측정값
0 = 원래 Missing이었던 값
```

## 4.2 이상 신호

0 대치 데이터로 ANOVA를 수행하자 일부 Feature에서 비정상적으로 강한 차이와 사실상 0에 가까운 p-value가 다수 나타났다.

Missing 비율이 Pass / Fail 집단에서 다르면, Sensor 값 자체의 차이가 아니라 **인위적으로 넣은 0의 개수 차이**가 평균 차이처럼 계산될 수 있었다.

## 4.3 수정

0 대치 데이터는 최종 분석에서 폐기하고 Raw Data로 복귀했다.

EDA에서는 NaN을 유지했다.

이 경험 이후 프로젝트의 전처리 원칙을 명시했다.

> **Missing ≠ Zero**

### 배운 점

전처리는 단순한 기술 단계가 아니라 **데이터 의미를 바꾸는 분석 의사결정**이다.

---

# 5. Phase 2 — Spotfire EDA

Spotfire에서는 다음을 확인했다.

- Pass / Fail 비율
- Feature별 Missing Rate
- Sensor Distribution
- Time Trend
- 월별 Fail Count와 Fail Rate

시간축에서 7~9월에 Fail 점이 많아 보였지만 월별 Sample 수가:

- 7월: 117
- 8월: 471
- 9월: 413

으로 해당 기간 자체의 관측량이 많았다.

따라서 Fail Count만 보고 특정 월을 이상 시기로 단정하지 않고 `Avg(Fail_Flag)`로 Fail Rate를 함께 확인했다.

> **분자는 반드시 분모와 함께 본다.**

---

# 6. Phase 3 — ANOVA Screening

590개 Sensor 각각에 대해 Pass / Fail 평균 차이를 One-way ANOVA로 확인했다.

590개의 동시 검정에서 False Positive를 줄이기 위해:

- BH-FDR
- Bonferroni

를 함께 적용했다.

결과:

- BH Significant: **36개**
- Bonferroni Significant: **15개**

Sensor 59가 가장 높은 효과크기를 보였다.

- F ≈ **38.757**
- p ≈ **6.16e-10**
- Eta² ≈ **0.02427**

그러나 가장 높은 Eta²도 약 2.4%였다.

즉:

> **Statistical Significance ≠ Practical Separation**

Spotfire Box Plot / Histogram에서도 Sensor 59의 Pass / Fail 분포는 상당 부분 중첩됐다.

---

# 7. 중요한 Leakage 결정 — ANOVA 36개를 ML에 바로 넣지 않음

전체 데이터 Label을 사용해 선정한 ANOVA 36개를 바로 ML Feature로 사용하면 편하지만, Test Sample의 Label 정보가 Feature Selection에 이미 사용된다.

```text
전체 데이터
↓
Label 기반 Feature Selection
↓
Train/Test Split
```

은 Test 정보를 간접적으로 본 셈이다.

따라서 전체 데이터 ANOVA 후보는 **EDA 후보군**으로만 사용했다.

ML은 다시 Raw 590 Feature에서 시작했다.

후속 Feature Selection 실험에서는 각 CV Fold의 Train 내부에서 ANOVA Rank를 새로 계산했다.

---

# 8. Phase 4 — Leakage-free Train / Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Train
- 1,253 Samples
- Pass: 1,170
- Fail: 83
- Fail Rate: 6.62%

### Test
- 314 Samples
- Pass: 293
- Fail: 21
- Fail Rate: 6.69%

Test는 이후 모델·전처리·Threshold가 최종 결정될 때까지 봉인했다.

---

# 9. Baseline — Accuracy의 함정 확인

첫 XGBoost Baseline은 Native NaN Handling + Unweighted였다.

초기 Test Diagnostic:

```text
TN = 292
FP = 1
FN = 21
TP = 0
```

Fail 21개를 **단 하나도 잡지 못했지만 Accuracy는 약 93%**였다.

이 결과가 Accuracy를 주요 기준으로 버린 직접적인 계기가 됐다.

> **Accuracy가 높아도 Fail Detection은 0일 수 있다.**

이후 모델 비교는 AP, Precision, Recall, F1, Confusion Matrix를 사용했다.

---

# 10. Missing Strategy 비교 — A0 / A1 / B0 / B1

모델 계열을 다음처럼 정의했다.

### A0
Native NaN + Unweighted XGBoost

### A1
Native NaN + Weighted XGBoost

### B0
Median Imputation + Missing Indicator + Unweighted XGBoost

### B1
Median Imputation + Missing Indicator + Weighted XGBoost

Median은 각 Fold Train에서만 fit했다.

Missing Indicator는:

```text
0 = 원래 값 존재
1 = 원래 Missing
```

을 보존한다.

## 10.1 5-Fold CV 결과

| Model | Precision Mean | Recall Mean | F1 Mean | AP Mean |
|---|---:|---:|---:|---:|
| A0 | 0.1000 | 0.0118 | 0.0211 | 0.1609 |
| A1 | 0.1000 | 0.0235 | 0.0381 | 0.1636 |
| B0 | **0.1667** | 0.0235 | 0.0411 | **0.1833** |
| B1 | 0.1500 | **0.0353** | **0.0571** | 0.1750 |

결론:

- Median + Missing Indicator가 상대적으로 유망했다.
- Class Weight는 Recall을 일부 개선했지만 여전히 매우 낮았다.
- 4/5 Fold에서 TP=0인 상황이 계속 나타났다.

따라서 Class Weight만으로는 불균형 문제를 충분히 해결하지 못한다고 판단했다.

## 10.2 Fold AP 평균과 pooled OOF AP 차이

후속 Phase에서는 모든 Fold의 OOF probability를 하나로 합친 뒤 AP를 다시 계산했다.

예:

| Model | Fold AP Mean | Pooled OOF AP |
|---|---:|---:|
| B0 | 0.1833 | 0.1632 |
| B1 | 0.1750 | 0.1561 |

이는 오류가 아니다.

> **Fold별 AP의 평균**과 **전체 OOF Prediction을 합쳐 한 번 계산한 AP**는 다른 통계량이다.

후속 모델 비교에서는 pooled OOF AP를 중심으로 사용했다.

---

# 11. B2 — SMOTENC 도입

## 11.1 왜 Oversampling을 검토했는가

Class Weight로도 실제 Fail 탐지가 거의 개선되지 않았다.

Train의 Fail은 83개뿐이고 Fold Train에서는 약 66~67개 수준이었다.

따라서 Minority Class 학습 기회를 늘리는 Oversampling을 검토했다.

## 11.2 왜 일반 SMOTE가 아니라 SMOTENC인가

Median + Missing Indicator 이후 입력은:

```text
590 continuous Sensor values
+
binary Missing Indicators
```

이다.

일반 SMOTE는 binary Indicator도 선형 보간해 `0.37`, `0.62` 같은 의미 없는 값을 만들 수 있다.

따라서 Missing Indicator를 categorical feature로 지정할 수 있는 **SMOTENC**를 선택했다.

## 11.3 왜 RobustScaler를 사용했는가

SMOTE 계열은 거리 기반으로 Minority 이웃을 선택한다.

Sensor별 Scale 차이가 크면 큰 값 범위의 Sensor가 거리 계산을 지배할 수 있다.

따라서:

1. Sensor 590개만 RobustScaler
2. Scaling 공간에서 SMOTENC 이웃 탐색
3. Synthetic Feature 생성
4. Sensor scale 원복
5. XGBoost 학습

순으로 구성했다.

EDA에서 Outlier가 많았기 때문에 RobustScaler를 사용했다.

## 11.4 적용 범위

SMOTENC는 **Fold Train에만** 적용했다.

```text
Fold Train
↓ Imputation
↓ Scaling
↓ SMOTENC
↓ XGBoost

Fold Validation
↓ Train-fitted Imputation only
↓ Prediction
```

Validation / Test는 원래 Class Distribution을 유지했다.

---

# 12. B2 초기 결과

초기 SMOTENC:

```text
sampling_strategy = auto (= 1:1)
k_neighbors = 5
```

Pooled OOF @ threshold 0.5:

| Model | TN | FP | FN | TP | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 1166 | 4 | 81 | 2 | 0.3333 | 0.0241 | 0.0449 | 0.1632 |
| B1 | 1163 | 7 | 80 | 3 | 0.3000 | 0.0361 | 0.0645 | 0.1561 |
| **B2** | **1155** | **15** | **75** | **8** | **0.3478** | **0.0964** | **0.1509** | **0.1769** |

SMOTENC는 FP를 늘렸지만:

- TP 2~3 → 8
- Recall 2~4% → 9.6%
- F1 상승
- AP 상승

을 보여 후속 개선 후보로 채택했다.

---

# 13. Threshold를 0.5에 고정하지 않음

B0/B1의 OOF probability를 threshold별로 분석하면서 같은 모델도 threshold를 낮추면 Fail 탐지는 늘지만 False Positive가 증가함을 확인했다.

예를 들어 B1은 threshold 0.20에서:

- Precision: 0.3158
- Recall: 0.1446
- F1: 0.1983

을 기록했다.

따라서 역할을 분리했다.

> **모델의 ranking 능력 → AP**  
> **운영 경고 기준 → Threshold**

Hyperparameter Search에서 threshold 0.5의 Recall 하나만 보고 모델을 선택하지 않았다.

---

# 14. XGBoost Hyperparameter Search

B2 구조를 유지하면서 XGBoost Parameter를 탐색했다.

초기 Random Search 최고 후보:

```text
subsample = 0.7
reg_lambda = 5.0
reg_alpha = 0.0
n_estimators = 300
min_child_weight = 5
max_depth = 3
learning_rate = 0.05
colsample_bytree = 0.5
```

OOF AP ≈ **0.1869**

```text
B2 초기 0.1769
↓
XGB tuning 0.1869
```

---

# 15. SMOTENC 자체 튜닝

초기 `1:1, k=5`는 임의 기본값이었다.

따라서:

### Sampling Ratio
- 0.25
- 0.50
- 0.75
- 1.00

### k_neighbors
- 3
- 5
- 7

총 12개 조합을 비교했다.

최고 결과:

```text
sampling_ratio = 1.00
k_neighbors = 3
```

- AP: **0.1987**
- Precision@0.5: 0.4054
- Recall@0.5: 0.1807
- F1@0.5: 0.2500

기존 `1.0 / k=5` AP 0.1869보다 개선됐다.

해석:

> 이 데이터에서는 Synthetic Fail 생성 시 비교적 가까운 소수 이웃(k=3)을 사용하는 설정이 더 유리했다.

단, Synthetic Sample을 실제 물리적 Wafer 상태로 해석하지 않았다.

---

# 16. Feature Selection 실험 — 예상과 반대 결과

## 16.1 왜 시도했는가

Train은:

- Samples 1,253
- Fail 83
- Sensor 590

으로 Minority Sample 수에 비해 Feature가 매우 많았다.

불필요한 Feature를 줄이면 일반화가 좋아질 가능성을 검토했다.

## 16.2 Leakage 방지

전체 데이터 ANOVA Rank를 가져오지 않고 각 Fold Train에서 새로 계산했다.

```text
Fold Train
↓
590 Feature ANOVA
↓
Train 내부 Rank
↓
Top-K 선택
↓
Validation 적용
```

비교:

- Top 25
- Top 50
- Top 100
- Top 200
- Top 400
- All 590

## 16.3 결과

| Top K | AP | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---:|---:|---:|---:|---:|
| **590** | **0.1987** | **0.4054** | 0.1807 | **0.2500** |
| 400 | 0.1754 | 0.3333 | 0.1446 | 0.2017 |
| 25 | 0.1681 | 0.2130 | **0.2771** | 0.2408 |
| 200 | 0.1670 | 0.2500 | 0.1566 | 0.1926 |
| 100 | 0.1655 | 0.2462 | 0.1928 | 0.2162 |
| 50 | 0.1363 | 0.2118 | 0.2169 | 0.2143 |

예상과 달리 **590개 전체가 가장 높은 AP**였다.

## 16.4 왜 Feature Selection을 채택하지 않았는가

ANOVA는 각 Feature를 독립적으로 보는 단변량 방법이다.

XGBoost는 threshold 기반 비선형 관계와 여러 Feature의 조건부 조합을 사용할 수 있다.

후속 분석에서 실제로:

- Sensor 419: ANOVA Eta Rank 310 / XGB Rank 2
- Sensor 95: ANOVA Eta Rank 58 / XGB Rank 1

같은 사례가 나타났다.

따라서 결과를:

> “Feature Selection이 실패했다”

가 아니라

> **“단변량 기준으로 Feature를 줄이는 것이 본 데이터의 다변량 예측 정보를 손실시켰을 가능성이 있다.”**

라고 해석했다.

단, 특정 상호작용이 원인이라고 증명한 것은 아니다.

---

# 17. Local XGBoost Fine-Tuning

590개 전체와 `SMOTENC ratio=1.0, k=3`을 고정하고 기존 최고점 주변을 탐색했다.

최고 Candidate:

```text
max_depth = 2
min_child_weight = 3
gamma = 0.50
subsample = 0.70
colsample_bytree = 0.50
reg_alpha = 0.10
reg_lambda = 5.0
learning_rate = 0.05
n_estimators = 300
```

결과:

- AP: **0.2050**
- Precision@0.5: 0.3158
- Recall@0.5: 0.2169
- F1@0.5: 0.2571

기존 0.1987보다 소폭 개선됐다.

더 얕고 규제가 있는 Tree가 유리했다.

---

# 18. Learning Rate × Number of Trees

Local Tuning 이후 구조를 고정하고:

- `learning_rate`
- `n_estimators`

만 추가 비교했다.

최고는 다시:

```text
learning_rate = 0.05
n_estimators = 300
```

AP **0.2050**이었다.

주변 후보와 차이가 크지 않아 추가적인 무한 탐색을 중단했다.

이유:

> Test leakage가 없더라도 같은 CV 결과를 반복해서 보고 최고 조합을 계속 고르면 **CV 자체에 대한 selection-induced optimism**이 발생할 수 있다.

---

# 19. 최종 Threshold 결정

최종 Candidate의 OOF Probability를 0.01 단위로 분석했다.

## F1 최대점 — Threshold 0.43

- TN 1117
- FP 53
- FN 60
- TP 23
- Precision **0.3026**
- Recall **0.2771**
- F1 **0.2893**

## Recall ≥ 40% — Threshold 0.28

- TN 1031
- FP 139
- FN 48
- TP 35
- Precision **0.2011**
- Recall **0.4217**
- F1 **0.2724**

## Recall ≥ 50% — Threshold 0.16

- FP 275
- TP 42
- Precision 0.1325
- Recall 0.5060
- F1 0.2100

Fail 절반을 잡으려면 False Alarm이 급격히 증가했다.

Test 이전에 다음 Threshold를 고정했다.

- `0.50` — 기본값 참고
- `0.43` — F1-oriented primary point
- `0.28` — Recall-oriented scenario

Test를 본 뒤 변경하지 않기로 했다.

---

# 20. Final Test — 현실 검증

Train 전체에 최종 Pipeline을 fit한 뒤 처음으로 Test를 평가했다.

- Test Samples: 314
- Fail: 21
- Fail Prevalence: **0.0669**

## Test AP

**0.1213**

OOF AP **0.2050** 대비 큰 폭으로 하락했다.

## Locked Threshold 결과

| Threshold | TN | FP | FN | TP | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 285 | 8 | 20 | 1 | 0.1111 | 0.0476 | 0.0667 |
| 0.43 | 281 | 12 | 19 | 2 | 0.1429 | 0.0952 | 0.1143 |
| 0.28 | 262 | 31 | 15 | 6 | 0.1622 | 0.2857 | 0.2069 |

---

# 21. Test 성능이 왜 떨어졌는가

단일 원인으로 단정하지 않았다.

## 21.1 Fail Sample 부족

Test Fail은 21개뿐이다.

```text
1 / 21 ≈ 4.76%p
```

Fail 한 개 차이만으로 Recall이 크게 움직인다.

## 21.2 High-dimensional / Low-minority 구조

Train에는 590 Sensor에 Fail 83개뿐이다.

Minority Sample에 비해 Feature가 매우 많아 안정적인 일반화 패턴을 학습하기 어렵다.

## 21.3 반복적인 CV 기반 Model Selection

Test는 보지 않았기 때문에 직접적인 Test Leakage는 없었다.

하지만:

- XGB Random Search
- SMOTENC tuning
- Feature Selection
- Local tuning
- LR × trees

등에서 같은 CV 결과를 반복적으로 관찰했다.

따라서 최종 0.2050은 새로운 데이터에서 재현될 성능보다 다소 낙관적으로 선택됐을 가능성이 있다.

이를 **selection-induced optimism / CV overfitting 가능성**으로 해석했다.

## 21.4 데이터 자체의 Signal 제한

ANOVA 최고 Eta²도 약 2.4%였고 주요 Sensor의 Pass / Fail 분포 역시 크게 중첩됐다.

강한 단일 판별 신호 자체가 부족할 가능성이 있다.

## 21.5 시간 / 공정 상태 변화 가능성

후속 Spotfire에서 Sensor 129의 특정 월 변화가 확인됐다.

Random Split만으로 모든 공정 상태 변화를 충분히 대표하지 못했을 가능성이 존재한다.

향후 Time-based Validation이 필요한 이유다.

---

# 22. Test 이후 프로젝트 방향 전환

Test를 확인한 뒤 가능한 선택지는 세 가지였다.

### A. Test를 보고 Parameter / Threshold / Feature를 다시 조정
→ 하지 않음. Test가 Validation이 되기 때문.

### B. 낮은 Test 성능을 숨기고 OOF만 강조
→ 하지 않음.

### C. 예측 모델의 한계를 인정하고 변수 탐색 도구로 제한적으로 활용
→ **채택**

따라서 프로젝트의 결론을:

> “고성능 불량 예측기를 개발했다”

에서

> **“예측 모델의 일반화 한계를 검증한 뒤 통계·ML·안정성·Spotfire 재검증을 결합해 공정 관리 우선순위 후보를 도출했다.”**

로 정리했다.

이 방향 전환은 프로젝트의 핵심 의사결정이다.

---

# 23. Feature Importance — 단일 모델 결과를 그대로 믿지 않음

최종 XGBoost Total Gain 상위:

| Rank | Sensor | Total Gain % |
|---:|---:|---:|
| 1 | 95 | 7.753 |
| 2 | 419 | 6.574 |
| 3 | 511 | 6.268 |
| 4 | 59 | 4.655 |
| 5 | 486 | 4.188 |
| 6 | 33 | 3.141 |
| 9 | 129 | 2.337 |

그러나 Test AP가 낮았기 때문에 Final Model Importance만으로 관리 우선순위를 정하는 것은 위험하다고 판단했다.

따라서 Fold별 중요도 안정성을 추가 검증했다.

---

# 24. Feature Importance Stability

5개 CV Fold 각각에서 동일한 최종 Pipeline을 다시 학습하고 Sensor Total Gain Rank를 계산했다.

| Sensor | Mean Rank | Best | Worst | Top20 Count |
|---:|---:|---:|---:|---:|
| 419 | **2.4** | 1 | 5 | **5/5** |
| 511 | 4.2 | 1 | 8 | 5/5 |
| 95 | 4.4 | 1 | 11 | 5/5 |
| 486 | 5.8 | 2 | 16 | 5/5 |
| 33 | 8.0 | 3 | 15 | 5/5 |
| 59 | 9.4 | 1 | 20 | 5/5 |
| 129 | 19.2 | 14 | 35 | 4/5 |

이 과정을 통해:

> “Final Model에서 우연히 높았던 Feature”

와

> **“Fold가 달라도 반복적으로 중요했던 Feature”**

를 구분했다.

---

# 25. ANOVA × XGBoost Evidence 비교

## Sensor 59

- ANOVA Eta Rank: **1**
- BH Significant
- Bonferroni Significant
- XGB Rank: **4**
- CV Top20: **5/5**

→ 통계 + ML + Stability가 모두 지지

## Sensor 419

- ANOVA Eta Rank: **310**
- p ≈ **0.573**
- BH / Bonferroni: False
- XGB Rank: **2**
- CV Mean Rank: **2.4**
- CV Top20: **5/5**

→ 단변량 평균 차이는 거의 없지만 ML에서 반복 활용

## Sensor 95

- ANOVA Eta Rank: **58**
- 보정 후 비유의
- XGB Rank: **1**
- CV Top20: **5/5**

→ Statistical Importance와 Predictive Importance가 다를 수 있음을 보여주는 사례

---

# 26. 최종 Tier 설계

단순 XGB Rank가 아니라 Evidence Type을 기준으로 분류했다.

## Tier 1 — Multi-evidence

### Sensor 59
- Bonferroni Significant
- XGB Rank 4
- Top20 5/5

### Sensor 33
- BH Significant
- XGB Rank 6
- Top20 5/5

### Sensor 129
- Bonferroni Significant
- XGB Rank 9
- Top20 4/5

## Tier 2 — Stable ML-only

대표 심층검토:

- Sensor 419
- Sensor 95

추가 안정 ML 후보:

- 511
- 486
- 31
- 111 등

## Tier 3 — Secondary

최종 모델에서는 Top20이었지만 Fold 간 변동성이 큰 변수.

예:

- Sensor 487: Worst Rank 523
- Sensor 560: Worst Rank 580

Final Model Rank 하나만 보고 우선순위를 높이지 않았다.

---

# 27. Spotfire Re-validation

Tier 후보를 다시 Raw Distribution / Time / Missing 관점에서 검토했다.

## 27.1 Sensor 59 — 가장 강한 Multi-evidence 후보

### Distribution
- Fail Median: **5.5223**
- Pass Median: **0.7223**
- Fail Mean: **8.5151**
- Pass Mean: **2.5635**

Fail 분포가 높은 값 방향으로 이동했지만 두 Class는 상당 부분 겹쳤다.

### Time
뚜렷한 장기 Drift는 보이지 않았다. 높은 값도 Pass에서 존재했다.

### 결론

> **통계적 차이 + ML 중요도 + CV 안정성이 모두 확인된 최우선 검토 후보**

단독 Threshold로 Fail을 판별할 정도의 분리력은 없다.

---

## 27.2 Sensor 33 — 단변량으로 거의 설명되지 않는 후보

### Distribution
- Fail Median: 8.8941
- Pass Median: 8.7662
- Fail Mean: 9.3682
- Pass Mean: 8.9313

Box / Histogram이 크게 겹쳤다.

### Time
뚜렷한 Drift 없음.

### Missing
- Fail: 0%
- Pass: 약 0.068%

사실상 Missing 차이 없음.

그런데:

- XGB Rank 6
- Top20 5/5

### 결론

단변량 값, 시간, Missing으로는 높은 ML 중요도를 설명하기 어려웠다.

따라서 **다변량 모델 내부에서 반복 활용된 후보**로 해석하되 구체적 상호작용을 원인이라고 단정하지 않았다.

---

## 27.3 Sensor 129 — 시간 효과 발견

전체 분포:

- Fail Median: 0
- Pass Median: -0.1419
- Fail Mean: -0.0826
- Pass Mean: -0.5880

처음에는 Fail이 전체적으로 높다고 해석할 수 있었다.

그러나 Time Scatter에서 9~10월 부근 낮은 Pass 군집이 나타났고 월별 Median을 확인하자:

- 대부분 월: Pass / Fail Median이 유사
- **9월 Pass Median ≈ -2.6**
- Fail은 기존 수준 유지

9월 Sample:

- Pass: **396**
- Fail: **17**

소표본 몇 개로 발생한 현상으로 보기 어렵다.

### Missing
- Fail: 0%
- Pass: 약 0.6%

### 결론

> 전체 Pass / Fail 차이가 불량 자체의 효과만이 아니라 **특정 시기의 공정 상태 변화와 혼재했을 가능성**이 있다.

Feature Importance를 인과로 해석하면 위험하다는 대표 사례가 됐다.

---

## 27.4 Sensor 419 — Stable ML-only

ANOVA:

- Eta Rank 310
- p ≈ 0.573

하지만:

- XGB Rank 2
- Mean Rank 2.4
- Top20 5/5

### Distribution
- `0` 근처 큰 Spike
- 100~1000 수준의 넓은 Positive 영역

Zero 비율:

- Fail: 약 **37~38%**
- Pass: 약 **46%**

0 여부 하나만으로 중요도를 설명할 정도의 극적인 차이는 아니었다.

### Time
0 군집과 양수 값이 기간 전반에 존재하며 명확한 장기 Drift 없음.

### 결론

> 평균 차이는 약하지만 구간적 분포 또는 다변량 구조에서 반복 활용됐을 가능성이 있는 Stable ML-only 후보

---

## 27.5 Sensor 95 — Stable ML-only

Box Plot:

- Fail Median: 0.0001
- Pass Median: 0
- Fail Mean: 8.365e-05
- Pass Mean: 5.834e-05

단변량 차이는 매우 작다.

하지만:

- XGB Rank 1
- Mean Rank 4.4
- Top20 5/5

Time Scatter에서는 값이 임의의 연속값보다 일정한 수준에 층층이 나타났다.

→ **Step-like / Discrete-looking structure**

뚜렷한 장기 Drift 없음.

실제 센서 분해능, 반올림, 공정 단계값 등 무엇 때문인지는 익명 데이터이므로 단정하지 않았다.

---

# 28. 최종 Priority

| Sensor | Type | 최종 해석 |
|---:|---|---|
| **59** | Multi-evidence | 통계·ML·CV가 반복 지지한 최우선 검토 후보 |
| **33** | Multi-evidence | 단변량 효과는 작지만 ML에서 안정적으로 활용 |
| **129** | Multi-evidence + Time | 시간에 따른 공정 상태 변화 혼재 가능 |
| **419** | Stable ML-only | 구조적 분포를 가진 안정적인 ML 후보 |
| **95** | Stable ML-only | Step-like 구조를 보이는 안정적인 ML 후보 |

이 순위는:

> **Failure Cause Ranking이 아니라 Review Priority**

이다.

---

# 29. 폐기하거나 채택하지 않은 접근

## 폐기 — Missing = 0
실제 0과 Missing 의미가 섞여 통계 왜곡.

## 미채택 — Class Weight 단독
Fail Recall 개선이 제한적.

## 미채택 — ANOVA Top-K Feature Selection
OOF AP 감소.

## 미채택 — Test 결과 기반 재튜닝
Test Leakage 위험.

## 미채택 — Feature Importance = 원인 해석
익명화 + 낮은 Test 성능 + 시간 효과 가능성.

## 미채택 — Tier 2/3 전체 Sensor 심층 Spotfire 분석
대표 사례만으로 방법론 검증이 충분했고 목적 대비 노동량이 과도하다고 판단.

---

# 30. 면접에서 반드시 설명할 수 있어야 하는 질문

## Q1. 왜 Accuracy를 사용하지 않았나?
Fail이 6.6%뿐이라 모두 Pass로 예측해도 약 93% Accuracy가 나오기 때문이다. AP, Precision, Recall, F1을 사용했다.

## Q2. 왜 Median Imputation을 사용했나?
Median을 최적의 물리적 보정법이라고 주장한 것이 아니다. 비교 가능한 단순 baseline으로 사용하고 Missing Indicator로 원래 결측 여부를 보존했다.

## Q3. 왜 SMOTE가 아니라 SMOTENC인가?
Missing Indicator가 binary이기 때문이다. 일반 SMOTE는 0.3, 0.7 같은 의미 없는 indicator 값을 만들 수 있어 categorical 처리가 가능한 SMOTENC를 사용했다.

## Q4. 왜 Test에는 SMOTENC를 하지 않았나?
Test는 실제 Class Distribution을 대표해야 한다. Synthetic Sample을 Test에 추가하면 일반화 성능을 평가할 수 없다.

## Q5. 왜 ANOVA 36개를 바로 ML Feature로 쓰지 않았나?
전체 Label을 이용해 고른 Feature를 사용하면 Test Label 정보가 Feature Selection에 반영되는 Leakage가 발생하기 때문이다.

## Q6. Feature Selection은 왜 채택하지 않았나?
Train-only ANOVA Top-K를 비교했지만 590개 전체가 가장 높은 AP를 기록했다. 단변량 평균 차이가 작은 Feature도 Tree의 비선형·조건부 구조에서 유용할 가능성이 있어 전체 Feature를 유지했다.

## Q7. 왜 OOF AP 0.205인데 Test는 0.121인가?
Fail Sample 부족, 590 Feature 대비 작은 Minority Sample, 반복적인 CV 기반 선택에 따른 selection-induced optimism, 제한적인 예측 신호, 시간적 공정 상태 변화 가능성을 함께 고려한다. 한 가지 원인으로 단정하지 않는다.

## Q8. 그러면 모델이 실패한 것 아닌가?
생산 적용용 자동 불량 분류기로는 성능이 부족하다. 그 한계를 인정하고 모델을 변수 후보 탐색 도구로 제한적으로 활용한 뒤 통계·CV Stability·Spotfire 재검증과 결합했다.

## Q9. 왜 Sensor 59가 최우선 후보인가?
ANOVA Eta Rank 1, Bonferroni Significant, XGB Rank 4, CV Top20 5/5, Spotfire에서 Fail 분포 상향이 반복적으로 확인됐기 때문이다.

## Q10. Sensor 419는 ANOVA 비유의인데 왜 중요한가?
ANOVA는 평균 차이를 주로 보지만 419는 XGBoost에서 Fold가 바뀌어도 계속 최상위였다. Spotfire에서도 0 spike + 넓은 양수 분포가 나타났다. 단변량 통계와 다변량 예측 중요도가 서로 다른 정보를 줄 수 있음을 보여준다.

## Q11. Sensor 129에서 무엇을 배웠나?
전체 데이터만 보면 Fail이 더 높아 보였지만 월별 재검증에서 9월 Pass 집단의 큰 하락이 발견됐다. 통계 유의성과 Feature Importance를 바로 공정 원인으로 연결하면 안 되고 시간·공정 상태를 함께 봐야 한다는 점을 확인했다.

---

# 31. 이 프로젝트에서 가장 중요한 사고 흐름

```text
잘못된 0 대치
↓
왜곡 발견
↓
Raw 복원

Accuracy 착시
↓
AP / Recall 중심 평가

Class Weight 한계
↓
SMOTENC

Feature를 줄이면 좋아질 것이라는 가설
↓
CV에서 반증
↓
전체 590 Feature 유지

OOF 성능 개선
↓
독립 Test 성능 하락
↓
생산 적용 모델 주장 철회

Feature Importance
↓
Fold Stability
↓
Spotfire 재검증

통계 + ML + 시간축
↓
관리 우선순위 후보
```

프로젝트의 핵심은 **처음 세운 가설을 끝까지 방어한 것이 아니라, 데이터가 반대 결과를 보여줄 때 분석 방향을 수정한 과정**에 있다.

---

# 32. 최종 한 문장

> **UCI SECOM의 590개 익명 공정 변수를 대상으로 Leakage-free 불량 예측 파이프라인을 구축하고, 독립 Test에서 일반화 한계를 확인한 뒤 통계적 차이·ML 중요도·CV 안정성·시간축 재검증을 결합해 추가 공정 검토가 필요한 변수 후보를 도출했다.**

---

# 33. 관련 파일

```text
02_notebooks/
├── 01_raw_data_audit.ipynb
├── 02_spotfire_eda_prep.ipynb
├── 03_anova_screening.ipynb
├── 04_ml_pipeline.ipynb
├── 05_xgboost_missing_strategy.ipynb
└── 06_model_evaluation.ipynb

03_output/
├── sensor_missing_summary.csv
├── anova_screening_summary.csv
├── anova_bh_candidates.csv
├── anova_xgb_comparison.csv
├── final_sensor_priority.csv
└── final_priority_sensor_summary.csv

04_spotfire/
├── SECOM_Phase2_EDA.dxp
└── SECOM_Final_Analysis.dxp
```

---

## 문서 역할 구분

- `README.md`  
  처음 보는 사람이 프로젝트를 빠르게 이해하기 위한 요약

- `PROJECT_LOG.md`  
  시행착오, 실험 결과, 의사결정 근거를 보존하는 상세 기록

- `02_notebooks/`  
  실제 분석 코드 및 재현 근거

- `04_spotfire/SECOM_Final_Analysis.dxp`  
  EDA 및 최종 후보 재검증 결과

- `06_portfolio/`  
  취업 지원용 프로젝트 포트폴리오
