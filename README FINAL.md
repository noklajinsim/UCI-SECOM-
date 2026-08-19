# UCI SECOM 기반 반도체 공정 불량 예측 및 관리 우선순위 변수 탐색

> **590개의 익명화된 반도체 공정 측정 변수에서 불량 예측에 기여하는 후보를 선별하고,
> 통계·머신러닝·시간축 재검증을 결합해 엔지니어가 우선적으로 확인할 관리 후보를 도출한 프로젝트**

---

## 1. Project Overview

반도체 양산 공정에서는 수많은 공정·계측 변수가 동시에 수집되지만,
모든 변수를 동일한 수준으로 관리하는 것은 현실적으로 어렵습니다.

본 프로젝트에서는 UCI SECOM 반도체 공정 데이터를 활용하여 다음 질문에 답하고자 했습니다.

> **"수백 개의 익명 공정 변수 가운데 어떤 변수를 우선적으로 확인해야 하는가?"**

단순히 높은 분류 정확도를 만드는 데 집중하기보다,

- 데이터 품질 점검
- Spotfire 기반 EDA
- 통계적 변수 선별
- Leakage-free 머신러닝 파이프라인
- 불균형 데이터 대응
- Feature Importance 및 CV 안정성 검증
- 시간축·분포 기반 재검증

을 연결하여 **불량 예측 기여도가 높은 변수 후보를 압축하고 관리 우선순위를 제안하는 것**을 목표로 했습니다.

> **Feature Importance는 인과관계를 의미하지 않습니다.**
> 본 프로젝트에서는 중요 변수를 불량의 원인이 아닌
> **추가적인 공정 검토가 필요한 관리 후보**로 해석했습니다.

---

## 2. Dataset

**UCI SECOM Dataset**

| 항목 | 값 |
|---|---:|
| Samples | 1,567 |
| Sensor Features | 590 |
| Pass | 1,463 (93.36%) |
| Fail | 104 (6.64%) |
| Total Missing Values | 41,951 |
| Overall Missing Rate | 4.52% |

Target은 다음과 같이 변환했습니다.

- `-1` → Pass (`0`)
- `1` → Fail (`1`)

Fail 비율이 약 **6.6%**에 불과하여 심한 클래스 불균형을 가진 데이터입니다.

---

## 3. Analysis Workflow

```text
Raw Data Audit
      ↓
Spotfire EDA
      ↓
ANOVA + Multiple Testing Correction
      ↓
Leakage-free Train/Test Split
      ↓
Missing / Imbalance Strategy Comparison
      ↓
SMOTENC + XGBoost
      ↓
Hyperparameter / Threshold Analysis
      ↓
Independent Test Evaluation
      ↓
Feature Importance + CV Stability
      ↓
Statistical × ML Evidence Comparison
      ↓
Spotfire Re-validation
      ↓
Priority Sensor Candidates
```

---

## 4. Data Audit & EDA

원본 데이터는 수정하지 않고 별도로 보존했습니다.

### Data Quality Check

**주요 결과:**

- 전체 결측률: 4.52%
- 결측률 90% 이상 Feature 존재
- Strict Constant Feature: 0개
- 관측값은 일정하지만 Missing을 포함하는 Feature: 116개
- 실제 0 값: 200,005개
- 0 값을 포함하는 Feature: 215개

초기 분석 과정에서 Missing을 일괄적으로 0으로 대체했을 때
통계검정 결과가 크게 왜곡되는 것을 확인했습니다.

따라서 이후 분석에서는

> Missing ≠ 0

이라는 원칙을 유지하고 Raw Data에서 다시 분석했습니다.

### Spotfire EDA

Spotfire를 이용해 다음을 확인했습니다.

- Pass / Fail 비율
- Feature별 Missing Pattern
- Sensor Distribution
- Time Trend
- 월별 Fail Rate

특정 시기의 Fail 개수가 많아 보여도
그 기간의 전체 Sample 수가 많을 수 있으므로,
단순 Count가 아니라 비율과 분모를 함께 확인했습니다.

---

## 5. Statistical Screening

590개 Sensor 각각에 대해 Pass / Fail 차이를 탐색하기 위해
One-way ANOVA를 수행했습니다.

590개의 동시 검정에서 발생할 수 있는 False Positive를 줄이기 위해

- BH-FDR
- Bonferroni Correction

을 함께 적용했습니다.

| 기준 | Significant Features |
|---|---:|
| BH-FDR | 36 |
| Bonferroni | 15 |

가장 높은 효과크기를 보인 변수는 **Sensor 59**였습니다.

- F-statistic ≈ 38.76
- p-value ≈ 6.16 × 10⁻¹⁰
- Eta² ≈ 0.0243

가장 높은 Eta²도 약 2.4% 수준이었기 때문에,

> Statistical Significance ≠ Practical Separation

이라는 점을 함께 고려했습니다.

또한 전체 데이터의 Label을 사용한 ANOVA 결과를
머신러닝 Feature Selection에 직접 사용하지 않았습니다.

이는 Test Label 정보가 Feature Selection 과정에 반영되는
**Data Leakage를 방지**하기 위함입니다.

---

## 6. Leakage-free Machine Learning Pipeline

먼저 Train / Test를 분리했습니다.

| Dataset | Samples | Fail |
|---|---:|---:|
| Train | 1,253 | 83 |
| Test | 314 | 21 |

```python
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

이후 다음 과정은 모두 Train 또는 Train 내부 CV에서만 수행했습니다.

- Imputation
- Oversampling
- Feature Selection
- Hyperparameter Tuning
- Threshold Selection

Test 데이터는 최종 모델과 Threshold가 결정될 때까지 사용하지 않았습니다.

### Missing Strategy

비교한 방법:

- XGBoost Native NaN Handling
- Median Imputation + Missing Indicator

Median + Missing Indicator 방식이 상대적으로 유망하여
후속 모델의 기본 전처리 방식으로 채택했습니다.

### Class Imbalance

비교한 방법:

- Baseline XGBoost
- scale_pos_weight
- SMOTENC

Missing Indicator가 Binary 변수이므로 일반 SMOTE 대신
**SMOTENC**를 사용했습니다.

또한 거리 기반 Oversampling에서 Sensor Scale 차이의 영향을 줄이기 위해
590개 Sensor 값에만 RobustScaler를 적용해 이웃을 탐색했습니다.

Validation / Test에는 Oversampling을 적용하지 않았습니다.

Synthetic Sample은 실제 Wafer 또는 물리적으로 검증된 공정 상태가 아니라
Feature Space상의 학습용 합성 데이터로만 해석했습니다.

---

## 7. Model Development

모델 선택의 주요 기준은
불균형 데이터에 적합한 **Average Precision(AP)**으로 설정했습니다.

### OOF Performance Improvement

```text
Initial SMOTENC + XGBoost
AP 0.1769
      ↓
XGBoost Parameter Search
AP 0.1869
      ↓
SMOTENC Ratio / Neighbor Tuning
AP 0.1987
      ↓
Local XGBoost Fine-Tuning
AP 0.2050
```

**최종 SMOTENC 설정:**

```python
sampling_strategy = 1.0
k_neighbors = 3
```

**최종 XGBoost 설정:**

```python
max_depth          = 2
min_child_weight   = 3
gamma              = 0.5

subsample          = 0.7
colsample_bytree   = 0.5

reg_alpha          = 0.1
reg_lambda         = 5.0

learning_rate      = 0.05
n_estimators       = 300
```

---

## 8. Feature Selection Experiment

ANOVA 기반 Feature Ranking을 각 CV Fold의 Train 내부에서 새로 계산하여
Leakage-free Top-K Feature Selection을 비교했습니다.

| Features | OOF AP |
|---:|---:|
| 25 | 0.1681 |
| 50 | 0.1363 |
| 100 | 0.1655 |
| 200 | 0.1670 |
| 400 | 0.1754 |
| 590 | 0.1987 |

단변량 통계 기준으로 Feature를 줄일수록 오히려 성능이 감소했습니다.

따라서 최종 모델에서는 590개 Sensor 전체를 유지했습니다.

이는 단변량 평균 차이만을 기준으로 변수를 제거할 경우,
Tree Model이 활용할 수 있는 비선형 관계 또는 변수 간 조합 정보가
손실될 가능성을 시사합니다.

---

## 9. Threshold Analysis

모델 자체의 비교에는 AP를 사용하고,
실제 분류 Threshold는 별도로 분석했습니다.

### F1-oriented Operating Point

Threshold = 0.43

| Precision | Recall | F1 |
|---:|---:|---:|
| 0.3026 | 0.2771 | 0.2893 |

### Recall-oriented Operating Point

Threshold = 0.28

| Precision | Recall | F1 |
|---:|---:|---:|
| 0.2011 | 0.4217 | 0.2724 |

실제 공정에서 False Negative와 False Positive의 비용 정보가 없기 때문에,
하나의 Threshold를 절대적 최적값으로 주장하지 않고

> Fail Detection과 False Alarm의 Trade-off

를 제시했습니다.

---

## 10. Independent Test Evaluation

모델 구조와 Threshold를 모두 Train에서 결정한 뒤
마지막으로 독립 Test 데이터를 평가했습니다.

**Test AP: 0.1213**

Test Fail prevalence는 약 0.0669였습니다.

| Threshold | Precision | Recall | F1 | TP | FP |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.1111 | 0.0476 | 0.0667 | 1 | 8 |
| 0.43 | 0.1429 | 0.0952 | 0.1143 | 2 | 12 |
| 0.28 | 0.1622 | 0.2857 | 0.2069 | 6 | 31 |

Train OOF AP 0.2050 대비 Test AP가 0.1213으로 감소했습니다.

따라서 본 모델을

> 생산 적용 수준의 안정적인 불량 분류기

로 해석하지 않았습니다.

Fail Sample 자체가 매우 적고,
Train CV 결과를 기반으로 반복적인 모델 선택을 수행했기 때문에
CV 결과에 대한 selection-induced optimism 가능성도 고려했습니다.

Test 결과 확인 후 모델이나 Threshold를 다시 조정하지 않았습니다.

---

## 11. Feature Importance & Stability

예측 성능의 한계를 인정한 상태에서,
모델이 어떤 Sensor를 반복적으로 활용했는지 분석했습니다.

**최종 XGBoost Total Gain 상위 변수:**

| Rank | Sensor |
|---:|---:|
| 1 | 95 |
| 2 | 419 |
| 3 | 511 |
| 4 | 59 |
| 5 | 486 |
| 6 | 33 |
| 9 | 129 |

단일 Final Model의 Importance만으로 판단하지 않고,
5-Fold 각각에서 Sensor 중요도 순위를 다시 계산해
Importance Stability를 확인했습니다.

| Sensor | Mean Rank | Top20 Count |
|---:|---:|---:|
| 419 | 2.4 | 5/5 |
| 511 | 4.2 | 5/5 |
| 95 | 4.4 | 5/5 |
| 486 | 5.8 | 5/5 |
| 33 | 8.0 | 5/5 |
| 59 | 9.4 | 5/5 |
| 129 | 19.2 | 4/5 |

---

## 12. Statistical × ML Evidence Comparison

ANOVA Effect Size와 XGBoost Importance를 비교한 결과,
두 기준은 항상 일치하지 않았습니다.

### Sensor 59 — Multi-evidence Candidate

- ANOVA Eta² Rank: 1
- Bonferroni Significant
- XGBoost Rank: 4
- CV Top20: 5/5

통계적 Pass/Fail 차이와 ML 예측 기여가 동시에 반복적으로 확인되었습니다.

### Sensor 419 — Stable ML-only Candidate

- ANOVA Eta² Rank: 310
- ANOVA Not Significant
- XGBoost Rank: 2
- CV Mean Rank: 2.4
- CV Top20: 5/5

평균 차이는 거의 없지만
다변량 XGBoost에서는 매우 안정적으로 활용되었습니다.

이를 통해

> Univariate Statistical Importance와
> Multivariate Predictive Importance는 서로 다른 정보를 제공할 수 있음

을 확인했습니다.

---

## 13. Spotfire Re-validation

통계 및 ML에서 도출한 후보들을
Spotfire에서 다시 분포·시간·Missing 관점으로 검토했습니다.

### Sensor 59

- Fail Median: 5.52 / Pass Median: 0.72
- Fail Mean: 8.52 / Pass Mean: 2.56

Fail 분포가 상대적으로 높은 값 영역으로 이동했지만,
두 Class의 분포는 상당 부분 중첩되었습니다.

→ 통계·ML·CV가 함께 지지하는 최우선 검토 후보

### Sensor 33

Pass / Fail 평균·중앙값 차이가 작고
Histogram, Time Trend, Missing Pattern에서도 뚜렷한 차이가 없었습니다.

그러나 XGBoost에서는 모든 Fold에서 Top20에 포함되었습니다.

→ 단변량보다는 다변량 예측 과정에서 반복적으로 활용된 후보

### Sensor 129

전체 데이터에서는 Fail 값이 상대적으로 높았지만,
시간축 분석에서 9월 Pass Median이 약 -2.6까지 급락했습니다.

9월 Sample: Pass 396 / Fail 17

따라서 통계적 Pass/Fail 차이에
시간에 따른 공정 상태 변화가 혼재했을 가능성을 고려했습니다.

### Sensor 419

ANOVA에서는 유의하지 않았지만
XGBoost에서는 매우 안정적인 중요 변수였습니다.

분포에서는 0 근처의 큰 Spike와 넓은 Positive Value 영역이 함께 나타났으며,
뚜렷한 장기 Time Drift는 확인되지 않았습니다.

→ 구간적·다변량 구조에서 활용되었을 가능성이 있는 Stable ML Candidate

### Sensor 95

단변량 효과는 작았지만 XGBoost Rank 1, CV Top20 5/5를 기록했습니다.

시간축에서는 일정한 값 단계가 반복되는
Step-like / Discrete Pattern이 관찰되었으며,
뚜렷한 장기 Drift는 없었습니다.

→ Stable Multivariate ML Candidate

---

## 14. Final Priority Candidates

최종 우선순위는 단순 XGBoost Rank가 아니라

- Statistical Evidence
- ML Importance
- CV Stability
- Spotfire Re-validation

을 함께 고려해 결정했습니다.

| Sensor | Evidence Type | Interpretation |
|---:|---|---|
| 59 | Multi-evidence | 통계·ML·CV에서 모두 반복 확인된 최우선 검토 후보 |
| 33 | Multi-evidence | 단변량 효과는 작지만 ML에서 안정적으로 활용 |
| 129 | Multi-evidence + Time | 시간에 따른 공정 상태 변화가 혼재된 후보 |
| 419 | Stable ML-only | 구조적 분포를 가진 안정적 다변량 ML 후보 |
| 95 | Stable ML-only | Step-like 구조를 보이는 안정적 ML 후보 |

이 Priority는 불량 원인 순위가 아니라 **Evidence Strength에 따른 검토 우선순위**입니다.

---

## 15. Key Findings

1. **Accuracy만으로는 불균형 데이터의 모델 성능을 평가할 수 없었습니다.**
   Fail 비율이 약 6.6%이므로 모든 Sample을 Pass라고 예측해도 약 93% Accuracy가 나옵니다. 따라서 Precision, Recall, F1, AP를 사용했습니다.

2. **Missing과 실제 0은 반드시 구분해야 했습니다.**
   Missing을 일괄 0으로 대체했을 때 통계검정 결과가 왜곡되었습니다.

3. **Statistical Significance와 ML Importance는 동일하지 않았습니다.**
   Sensor 59처럼 두 기준에서 모두 강한 변수도 있었지만, Sensor 419·95처럼 단변량 통계에서는 약하지만 ML에서 반복적으로 활용된 변수도 존재했습니다.

4. **Feature Importance는 원인을 의미하지 않았습니다.**
   Sensor 129에서는 시간 변화가 통계적 차이에 섞여 있을 가능성이 확인되었습니다.

5. **독립 Test 검증은 모델의 한계를 드러냈습니다.**
   OOF에서는 성능을 개선했지만 Test 일반화 성능은 낮았습니다. 이를 숨기거나 Test 결과에 맞춰 모델을 다시 조정하지 않았습니다.

---

## 16. Limitations

- Sensor 의미가 익명화되어 있어 물리적 공정 해석에 한계가 있음
- Fail Sample이 전체 104개로 매우 적음
- Final Test Fail은 21개로 지표 변동성이 큼
- SMOTENC Synthetic Sample은 실제 Wafer 상태를 의미하지 않음
- Final Test AP는 0.1213으로 생산 적용 수준의 성능을 확보하지 못함
- 반복적인 CV 기반 모델 선택 과정에서 selection-induced optimism이 발생했을 가능성
- 실제 Equipment / Recipe / Lot / Process Step Metadata가 없어 원인 검증이 불가능함

따라서 본 프로젝트의 최종 산출물을 자동 불량 판정 모델이 아니라

> **데이터 기반 공정 변수 관리 우선순위 탐색 방법론**

으로 해석했습니다.

---

## 17. Repository Structure

```text
UCI-SECOM/
│
├── 01_raw/
│   └── uci-secom.csv
│
├── 02_notebooks/
│   ├── 01_raw_data_audit.ipynb
│   ├── 02_spotfire_eda_prep.ipynb
│   ├── 03_anova_screening.ipynb
│   ├── 04_ml_pipeline.ipynb
│   ├── 05_xgboost_missing_strategy.ipynb
│   └── 06_model_evaluation.ipynb
│
├── 03_output/
│   ├── sensor_missing_summary.csv
│   ├── anova_screening_summary.csv
│   ├── anova_bh_candidates.csv
│   ├── anova_xgb_comparison.csv
│   ├── final_sensor_priority.csv
│   └── final_priority_sensor_summary.csv
│
├── 04_spotfire/
│   ├── SECOM_Phase2_EDA.dxp
│   └── SECOM_Final_Analysis.dxp
│
├── 05_figures/
├── 06_portfolio/
│
├── .gitignore
└── README.md
```

---

## 18. Tech Stack

**Data Analysis**

- Python
- Pandas / NumPy
- SciPy / Statsmodels
- Scikit-learn
- imbalanced-learn
- XGBoost

**Visualization**

- Spotfire

**Version Control**

- Git
- GitHub

---

## 19. Future Work

실제 공정 Metadata가 확보된다면 다음 분석으로 확장할 수 있습니다.

- Sensor ↔ Process Step Mapping
- Equipment / Recipe / Lot 단위 분석
- Time-based Validation
- Drift Detection
- Sensor Interaction Analysis
- SHAP 기반 모델 해석
- False Negative / False Positive 비용을 반영한 Threshold Optimization

---

## Conclusion

본 프로젝트에서는 높은 모델 점수 자체보다
분석 과정의 신뢰성과 공정 관점의 해석 가능성에 초점을 맞췄습니다.

독립 Test에서 일반화 성능의 한계를 확인했지만,
Test에 맞춰 결과를 재조정하지 않고

> 통계 분석 → 머신러닝 → CV 안정성 → Spotfire 재검증

을 연결해 590개의 익명 공정 변수에서
추가 검토가 필요한 관리 후보를 압축했습니다.

불량을 예측하는 것에서 끝나는 것이 아니라,
**엔지니어가 어디부터 확인해야 하는지를 제시하는 것.**

이것을 본 프로젝트의 최종 목표로 삼았습니다.
