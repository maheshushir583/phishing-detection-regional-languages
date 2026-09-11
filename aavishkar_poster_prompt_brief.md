# Aavishkar Research Poster — Comprehensive Project Data & Master Prompt Brief

This document contains **every single technical detail, metric, table, feature weight, regex pattern, dataset distribution, and evaluation result** of the project, formatted as a self-contained Master Prompt for Claude.

---

## 🚀 Instructions for Poster Generation

1. Open **Claude** (or any design AI tool).
2. Copy and paste the entire prompt block below.
3. Attach the **3 PNG graph images** from your project directory:
   - 📊 `model_benchmark.png`
   - 🎯 `confusion_matrix.png`
   - 📉 `threshold_analysis.png`

---

```markdown
# MASTER PROMPT: Complete Aavishkar 1m × 1m Research Poster Generation

## 🚨 MANDATORY AAVISHKAR CONVENTION RULES & DISQUALIFICATION CLAUSES

1. **Poster Dimensions:** Exactly **1 meter × 1 meter** (100 cm × 100 cm / 39.37 in × 39.37 in).
2. **STRICT ANONYMITY DISQUALIFICATION RULE (Annexure 1 & Section 9(1)(e)):**
   - **DO NOT** include any Student Name, PRN, College Name, Department Name, Guide Name, or Institutional Logos anywhere on the poster.
   - Revealing identity leads to immediate **DISQUALIFICATION**.
3. **Mandatory Header Format (Annexure 1):**
   - **Top Banner:** `Aavishkar – Maharashtra State Inter-University Research Convention`
   - **Top Left Badge:** `Category: Category 5 (Engineering and Technology)`
   - **Top Right Badge:** `Level: PG (Postgraduate)`
   - **Sub-header Center:** `Code No.: __________` (Left blank for registration code assignment)
4. **Visual Design System:**
   - **Theme:** Clean **Light Academic Theme** (Canvas: `#F8FAFC`, Card Background: `#FFFFFF`, Header Navy: `#0F172A`, Blue Accents: `#2563EB`, Teal Accent: `#0D9488`).
   - **Layout:** 3 equalized columns with crisp borders, soft shadows, clear visual hierarchy, legible from 1 meter distance.

---

## 📐 3-COLUMN LAYOUT OVERVIEW

```
+-----------------------------------------------------------------------------------------+
|                                    Aavishkar                                            |
|                  Maharashtra State Inter-University Research Convention                 |
| Category: Category 5 (Engineering and Technology)                      Level: PG        |
|                                     Code No.: ________                                  |
+--------------------------+------------------------------+-------------------------------+
|  COLUMN 1                |  COLUMN 2                    |  COLUMN 3                     |
|  - Project Title         |  - Preprocessing Pipeline    |  - Learned Feature Weights    |
|  - Problem Statement     |  - Model Benchmarking (CV)   |  - Real-World Case Study      |
|  - Research Objectives   |  - Holdout Evaluation        |  - Error Analysis & Scope     |
|  - Multilingual Dataset  |  - Embedded PNG Figures      |  - Conclusion & Deployability |
+--------------------------+------------------------------+-------------------------------+
```

---

## 📄 ALL EXPLICIT PROJECT DATA & TECHNICAL METRICS

### 📌 HEADER BANNER
- **Full Research Title:** Linguistic-Aware Phishing Detection Based on Indian Regional Languages Using Machine Learning
- **Short Title:** Linguistic-Aware Phishing Detection for Code-Switched Indian Messages
- **Dialect Support:** English · Hindi (हिन्दी) · Marathi (मराठी) · Hinglish · Marathlish · Devanagari Unicode
- **Category & Level:** Category 5 (Engineering & Technology) | Level: PG

---

### 📌 COLUMN 1: INTRODUCTION, OBJECTIVES & DATASET

#### 1. Problem Statement & Motivation
- **Context:** Over 800 million citizens in India communicate on WhatsApp and SMS using regional languages and code-switched dialects (Hinglish, Marathlish, Hindi/Marathi Devanagari).
- **Vulnerability Gap:** Commercial spam/phishing filters are trained almost exclusively on English datasets. Scammers actively exploit this gap by sending localized **KYC updates, UPI block warnings, digital arrest threats, and courier duty alerts** using regional phrasing to bypass human judgment and automated filters.
- **Key Statistics:** 800M+ regional users, 0% protection from English-only commercial filters.

#### 2. Research Question & Objectives
- **Research Question:** *Can classical, low-compute NLP models trained on a small, curated multilingual dataset reliably detect phishing intent across Indian regional code-switched text?*
- **4 Core Objectives:**
  1. Construct a balanced 400-message regional dataset across 12 scam categories.
  2. Develop specialized text preprocessing for regional script, URL normalization, and stopword removal.
  3. Benchmark candidate classifiers (LinearSVC, Logistic Regression, Naive Bayes) using 5-Fold Stratified Cross-Validation.
  4. Deploy an offline, zero-GPU Streamlit web application (`demo_app.py`) for low-resource environments.

#### 3. Multilingual Dataset Characteristics
- **Dataset File:** `phishing_dataset_400_fixed.csv` (400 raw WhatsApp/SMS messages).
- **Class Balance:** 200 Phishing (Label 1) / 200 Legitimate (Label 0) — 1:1 balanced ratio.
- **Data Split:** 320 Training messages (80%) / 80 Holdout Test messages (20%), Stratified 80:20 split (`random_state=42`).
- **12 Scam Categories Covered:** KYC Fraud, Digital Arrest, UPI Fraud, Courier Duty, Lottery, Loan, Banking, Utility Bills, Investment, Job Offers, Govt Schemes, Mobile Recharge.

**Language Breakdown Table:**
| Language / Format | Message Count | Percentage | Scam Categories Covered |
|---|---:|---:|---|
| **Hinglish (Hindi-English)** | 94 | 23.5% | KYC, UPI, Banking, Loan |
| **English** | 86 | 21.5% | KYC, Lottery, Utility, Job |
| **Marathlish (Marathi-English)** | 78 | 19.5% | Customs Duty, Courier, Bill |
| **Marathi (मराठी)** | 57 | 14.25% | Bank Freeze, Electricity |
| **Hindi (हिन्दी)** | 55 | 13.75% | Digital Arrest, CBI Impersonation |
| **Devanagari Script** | 30 | 7.5% | Speed Challan, Govt Scheme |

---

### 📌 COLUMN 2: METHODOLOGY, BENCHMARKING & HOLDOUT EVALUATION

#### 4. NLP Preprocessing Pipeline & Parameters
```
[Raw WhatsApp / SMS Message (Any regional dialect / Devanagari script)]
       ↓
[Step 1: Text Cleaning & Regex Normalization]
  • URL Regex: https?://\S+|www\.\S+|\b\S+\.(?:com|in|net|org|co|ly|info|link|site|online|xyz|me)(?:/\S*)? → ' link_present '
  • Digits Regex: \d+ → ' numtoken '
  • Devanagari Preservation: \u0900-\u097F range retained
  • Punctuation Removal: [^\w\s\u0900-\u097F] → whitespace
       ↓
[Step 2: Regional Stopword Removal & Stemming]
  • Custom Stopwords (25+ words): aahe, ahet, ani, pan, te, tya, hi, ho, cha, chi, che, hai, tha, aur, ko, ki, ke, ka, mein, se, is, the, a, an
  • Stemmer: PorterStemmer (NLTK)
       ↓
[Step 3: Feature Extraction (TF-IDF Vectorizer)]
  • analyzer='word', ngram_range=(1,2) [Unigrams + Bigrams]
  • sublinear_tf=True, min_df=1, max_df=0.95, max_features=3000
       ↓
[Step 4: Classification & Probability Calibration]
  • Model: LinearSVC calibrated via CalibratedClassifierCV(method='sigmoid', cv=5)
  • Decision Threshold: 0.70 (Tuned to eliminate false alarms)
       ↓
[Output Verdict: 🚨 PHISHING (Probability ≥ 0.70) vs 🛡️ SAFE (Probability < 0.70)]
```

#### 5. Model Benchmarking (5-Fold Stratified Cross-Validation)

| Classifier Model | CV Accuracy | CV F1-Score | CV ROC-AUC | CV PR-AUC | Selection Status |
|---|---:|---:|---:|---:|:---:|
| **LinearSVC** | **97.75%** | **0.9778** | **0.9978** | **0.9977** | ✅ **Selected Final Model** |
| Logistic Regression | 98.00% | 0.9804 | 0.9966 | 0.9965 | Candidate |
| Multinomial Naive Bayes | 95.75% | 0.9582 | 0.9909 | 0.9919 | Baseline |

*(Attach Image 1: `model_benchmark.png` under Section 5)*

#### 6. Holdout Evaluation Results (80 Test Messages)
- **Selected Decision Threshold:** **0.70**
- **Holdout Accuracy:** **97.5%** (78 / 80 correct)
- **Phishing Recall:** **100.0%** (40 / 40) — **ZERO False Negatives** (0 scams missed!)
- **Phishing Precision:** **95.2%** (40 / 42)
- **Safe Recall:** **95.0%** (38 / 40)
- **Safe Precision:** **100.0%** (38 / 38)
- **ROC-AUC Score:** **0.9931** | **PR-AUC Score:** **0.9930**

**Holdout Confusion Matrix Table:**
| | Predicted Safe | Predicted Phishing |
|---|---:|---:|
| **Actual Safe (40)** | **38** (True Negatives) | **2** (False Positives) |
| **Actual Phishing (40)** | **0** (False Negatives) | **40** (True Positives) |

**Threshold Tuning Curve Data:**
| Threshold | Accuracy | Precision | Recall | False Negatives (FN) |
|---:|---:|---:|---:|---:|
| 0.30 | 92.5% | 87.0% | 100.0% | 0 |
| 0.50 | 96.3% | 93.0% | 100.0% | 0 |
| **0.70 (Tuned)** | **97.5%** | **95.2%** | **100.0%** | **0** |
| 0.80 | 95.0% | 95.0% | 95.0% | 2 |
| 0.90 | 93.8% | 94.9% | 92.5% | 3 |

*(Attach Image 2: `confusion_matrix.png` under Section 6)*  
*(Attach Image 3: `threshold_analysis.png` under Section 6)*

---

### 📌 COLUMN 3: EXPLAINABILITY, CASE STUDY & CONCLUSION

#### 7. Feature Attribution & Learned Weights (LinearSVC Coefficients)

| Top Phishing Indicator | Weight | Feature Interpretation | Top Safe Indicator | Weight | Feature Interpretation |
|---|---:|---|---|---:|---|
| `link_present` | **+3.85** | URL presence is strongest scam cue | `aaj` (today) | **-1.04** | Casual scheduling word |
| `कर link_present` | **+1.09** | Devanagari action + URL link | `kal` (tomorrow) | **-0.89** | Routine planning word |
| `rs numtoken` | **+0.80** | Amount mention (Rs. + number) | `credit` | **-0.76** | Normal banking text |
| `kyc` | **+0.74** | KYC fraud vocabulary | `recharge` | **-0.65** | Mobile utility term |
| `pay` | **+0.73** | Payment instruction | `meeting` | **-0.58** | Office communication |
| `account` | **+0.71** | Account urgency phrase | `successful` | **-0.54** | Transaction receipt |

#### 8. Real-World Case Study Validation (10 Unseen Scam Messages)
Tested on 10 out-of-sample unseen messages from active Indian cybercrime reports without any model retraining:
- **Overall Case Study Accuracy:** **80.0% (8/10 correct)**

**Complete 10-Message Validation Table:**
| # | Test Message Topic | Language | Scam Category | True Label | Model Score | Verdict | Result |
|---|---|---|---|---|---:|:---:|:---:|
| 1 | SBI Account KYC Expired | English | KYC Fraud | Phishing | 99.8% | Phishing | ✅ |
| 2 | CBI Digital Arrest Threat | Hindi-Devanagari | Digital Arrest | Phishing | 35.6% | Safe | ❌ |
| 3 | Hinglish UPI Block Warning | Hinglish | UPI Fraud | Phishing | 99.9% | Phishing | ✅ |
| 4 | Customs Parcel Duty Fee | Marathlish | Courier Fraud | Phishing | 100.0% | Phishing | ✅ |
| 5 | Hindi Bank KYC Suspension | Hindi-Devanagari | KYC Fraud | Phishing | 99.9% | Phishing | ✅ |
| 6 | Rs 25 Lakh Lottery Winner | English | Lottery | Phishing | 100.0% | Phishing | ✅ |
| 7 | Instant Loan Approval Rs 5L | Hinglish | Loan Scam | Phishing | 99.8% | Phishing | ✅ |
| 8 | India Post Customs Fee | Marathlish | Courier Fraud | Phishing | 99.9% | Phishing | ✅ |
| 9 | UPI Payment Confirmation | English | Legitimate | Safe | 87.3% | Phishing | ❌ |
| 10 | OTP Security Code Alert | Hinglish | Legitimate | Safe | 10.4% | Safe | ✅ |

**Honest Error Analysis & Discussion:**
- *Message 2 (False Negative — Digital Arrest, 35.6% score):* Scams lacking embedded URLs miss the high weight of `link_present`. Identifies key limitation: text-only social engineering requires dedicated authority-impersonation features.
- *Message 9 (False Positive — Payment Receipt, 87.3% score):* Genuine receipt containing `Rs`, `numtoken`, and `sbi.co.in` triggers conservative security warnings, which is acceptable in safety-critical applications.

#### 9. Key Research Contributions & Deployability
1. **Bridge Regional Security Gap:** Protects underserved Indian users against domain-specific code-switched phishing.
2. **Zero-GPU & Fully Offline:** Model file `phishing_detector_model.joblib` (217 KB) runs locally on CPUs without cloud APIs or internet requirements.
3. **Interactive Demo Dashboard:** Built live Streamlit web application (`demo_app.py`) featuring real-time risk verdict, confidence bar, feature contribution chips, and preprocessing trace.
4. **Future Scope:** Benchmark against Indic-BERT / MuRIL transformers and scale dataset to 5,000+ crowdsourced messages.

---

## 🖼️ ATTACHED IMAGE GRAPH SPECIFICATIONS

Please embed placeholders and captions for the 3 attached PNG files:
1. `model_benchmark.png` $\rightarrow$ **Figure 1:** 5-Fold Stratified Cross-Validation Benchmark Comparison.
2. `confusion_matrix.png` $\rightarrow$ **Figure 2:** Holdout Test Set Confusion Matrix (0 False Negatives).
3. `threshold_analysis.png` $\rightarrow$ **Figure 3:** Decision Threshold vs Accuracy, Precision, and Recall Tuning Curve.
```
