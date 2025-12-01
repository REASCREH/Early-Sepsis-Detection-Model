
# Project Title🏥 Pediatric Sepsis Early Detection - Machine Learning Solution

What is Pediatric Sepsis?

Pediatric sepsis is a life-threatening medical emergency where a child's body has an overwhelming response to infection, leading to tissue damage, organ failure, and potentially death. It progresses rapidly and requires immediate intervention.





## 🚀 The Challenge

Early Detection is Critical: For every hour that sepsis treatment is delayed, mortality increases by 4-8%. Traditional clinical diagnosis often comes too late due to:

Non-specific symptoms in children

Rapid disease progression

Varied clinical presentations



## Hackathon Objective



Build a machine learning model that can predict sepsis onset 6 hours before clinical recognition, giving healthcare providers a crucial early warning window for timely intervention.


## Why This Matters


Global Health Crisis: Sepsis affects millions of children worldwide with high mortality rates

Time-Sensitive Treatment: Antibiotics and supportive care are most effective when administered early

Preventable Deaths: Early detection could save thousands of pediatric lives annually




## 📊 Dataset Description


Primary Data Files
SepsisLabel_*.csv - Core dataset containing:

person_id: Unique patient identifier

measurement_datetime: Timestamp of measurement

SepsisLabel: Binary target (0=No Sepsis, 1=Sepsis)

drugsexposure_*.csv - Medication administration data:

drug_concept_id: Medication identifier

route_concept_id: Administration route (IV, oral, etc.)

drug_datetime_hourly: Timestamp of drug administration


## Data Characteristics

Total Records: 331,639 time points

Unique Patients: 2,649 children

Class Distribution:

Sepsis Positive: 6,874 records (2.07%)

Sepsis Negative: 324,765 records (97.93%)

Severe Class Imbalance: Critical challenge requiring specialized handling





## 🏗️ Solution Architecture


1. Data Preprocessing Framework

Temporal Feature Extraction
We transform raw timestamps into clinically meaningful temporal patterns:

Day of Week Patterns: Capturing weekly hospital workflow variations

Hour of Day Cycles: Identifying diurnal physiological rhythms

Seasonal Trends: Accounting for infection rate variations throughout the year

Drug Exposure Integration
Medication data is processed to create patient-specific drug profiles:

Daily drug administration summaries

Administration route patterns

Temporal sequencing of treatments

2. Addressing Class Imbalance

Undersampling Strategy
Given the severe 98:2 class imbalance, we implement strategic undersampling:

Random reduction of non-sepsis cases to match sepsis case count

Preservation of temporal and patient-specific patterns

Prevention of model bias toward majority class

Why This Works

Forces the model to learn sepsis-specific signatures

Prevents trivial "always predict no sepsis" solutions

Improves sensitivity to rare but critical events

3. Advanced Feature Engineering

TF-IDF Vectorization for Drug Data
We treat daily drug administrations as "documents" and apply:

Term Frequency Analysis: Which drugs are administered frequently

Inverse Document Frequency: Which drugs are rare but significant

Multi-dimensional Embedding: Creating 200 informative features

Clinical Rationale

Rare antibiotics may signal serious infections

Drug combinations indicate treatment protocols

Administration timing reveals clinical urgency

4. Model Selection Rationale

Why XGBoost?
We selected Extreme Gradient Boosting for its:

Proven Performance with tabular medical data

Native Categorical Support for clinical codes

Built-in Regularization preventing overfitting

Interpretability Features for clinical validation

Computational Efficiency for potential real-time deployment

Model Configuration
The model is tuned for medical prediction tasks:

Conservative learning rate for stable convergence

Depth limiting to prevent over-complexity

Feature sampling for robustness

AUC optimization focusing on ranking ability

5. Validation Philosophy

Stratified Group K-Fold Cross-Validation
Our validation strategy ensures:

Patient Independence: Same patient never appears in both training and validation

Class Balance Preservation: Each fold maintains realistic sepsis prevalence

Temporal Integrity: Sequential patient data remains unbroken

Why This Matters

Mimics real-world deployment scenarios

Prevents optimistic bias from data leakage

Provides reliable performance estimates


## 📈 Performance Results


Cross-Validation Metrics

Consistent Performance Across Folds
The model demonstrates robust performance with:

Validation Accuracy: 91-96% across all folds

F1 Scores: 91-96% indicating balanced precision and recall

Training Consistency: Minimal overfitting observed

Primary Evaluation Metric

PR-AUC: 0.9675 (Area Under Precision-Recall Curve)

Why PR-AUC?: For imbalanced medical problems, PR-AUC provides a more realistic performance measure than traditional ROC-AUC, focusing specifically on the minority class (sepsis cases)

Clinical Interpretation of Results

High Precision-Recall Balance

Recall (Sensitivity): Model effectively identifies true sepsis cases

Precision (Specificity): Minimizes false alarms that could cause alert fatigue

Clinical Utility: Balances the need to catch all sepsis cases with the reality of clinical workflow constraints
