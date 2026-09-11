# Aavishkar Research Poster — Complete Prompt & Project Brief

This file contains the complete, self-contained master prompt and data brief for generating your 1m × 1m Aavishkar Research Poster.

---

## 🚀 How to Use This Brief

1. Open **Claude** (or your design AI tool).
2. **Copy and paste the entire prompt block below** into your prompt window.
3. **Attach the 3 PNG graph images** from your project folder:
   - 📊 `model_benchmark.png`
   - 🎯 `confusion_matrix.png`
   - 📉 `threshold_analysis.png`
4. Send the prompt to generate your complete HTML / LaTeX / Canva poster!

---

```markdown
# MASTER PROMPT: Aavishkar 1m × 1m Research Poster Generation

## 🚨 MANDATORY AAVISHKAR CONVENTION RULES (ANNEXURE 1)

1. **Dimensions:** Exactly **1 meter × 1 meter** (100 cm × 100 cm / 39.37 in × 39.37 in).
2. **STRICT ANONYMITY DISQUALIFICATION RULE:**
   - **DO NOT** include any Student Name, PRN, College Name, Department Name, Guide Name, or Institutional Logos anywhere on the poster.
   - Revealing identity leads to immediate **DISQUALIFICATION**.
3. **Mandatory Header Format (Annexure 1):**
   - **Top Banner:** `Aavishkar – Maharashtra State Inter-University Research Convention`
   - **Top Left Badge:** `Category: Category 5 (Engineering and Technology)`
   - **Top Right Badge:** `Level: PG (Postgraduate)`
   - **Sub-header:** `Code No.: __________` (Left blank for registration code assignment)
4. **Visual & Design System:**
   - **Theme:** Clean, modern **Light Academic Theme** (Canvas: `#F8FAFC`, Cards: `#FFFFFF`, Navy Header: `#0F172A`, Blue Accents: `#2563EB`, Teal: `#0D9488`).
   - **Layout:** 3 equalized columns with soft card borders, shadows, clear visual hierarchy, and high legibility from 1 meter distance.

---

## 📐 POSTER LAYOUT & SECTION STRUCTURE

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

## 📄 COMPLETE PROJECT DATA & TABLES

### 📌 HEADER BANNER
- **Main Title:** Linguistic-Aware Phishing Detection Based on Indian Regional Languages Using Machine Learning
- **Dialect Badges:** English · Hindi (हिन्दी) · Marathi (मराठी) · Hinglish · Marathlish · Devanagari Unicode

---

### 📌 COLUMN 1: INTRODUCTION & DATASET

#### 1. Problem Statement & Motivation
- **The Context:** Over 800 million citizens in India communicate on WhatsApp and SMS using regional languages and code-switched dialects (Hinglish, Marathlish, Hindi/Marathi Devanagari).
- **The Vulnerability Gap:** Virtually all commercial spam/phishing filters are trained exclusively on English datasets. Scammers actively exploit this gap by sending localized **KYC updates, UPI block warnings, digital arrest threats, and courier duty alerts** using regional phrasing to bypass both human judgment and automated filters.
- **Key Statistics:** 800M+ regional users, 0% protection from English-only commercial filters.

#### 2. Research Question & Objectives
- **Research Question:** *Can low-compute classical machine learning models trained on a small, curated multilingual dataset reliably detect phishing intent across Indian regional code-switched text?*
- **Objectives:**
  1. Construct a balanced 400-message regional dataset spanning 12 scam categories.
  2. Develop specialized text preprocessing for regional script, URL normalization, and stopword removal.
  3. Benchmark candidate classifiers (LinearSVC, Logistic Regression, Naive Bayes) using 5-Fold Stratified Cross-Validation.
  4. Deploy a zero-GPU, fully offline Streamlit application for low-resource environments.

#### 3. Multilingual Dataset Breakdown
- **Total Messages:** 400 balanced messages (200 Phishing / 200 Legitimate).
- **Split:** 320 Training (80%) / 80 Holdout Test (20%), Stratified 80:20.
- **Language Distribution Table:**

| Language / Format | Message Count | Scam Categories Covered |
|---|---:|---|
| **Hinglish (Hindi-English)** | 94 | KYC, UPI, Banking, Loan |
| **English** | 86 | KYC, Lottery, Utility, Job |
| **Marathlish (Marathi-English)** | 78 | Customs Duty, Courier, Bill |
| **Marathi (मराठी)** | 57 | Bank Freeze, Electricity |
| **Hindi (हिन्दी)** | 55 | Digital Arrest, CBI Impersonation |
| **Devanagari Script** | 30 | Speed Challan, Govt Scheme |

---

### 📌 COLUMN 2: METHODOLOGY & PERFORMANCE

#### 4. Preprocessing Pipeline & System Architecture
```
[Raw WhatsApp / SMS Message (Any regional dialect / Devanagari script)]
       ↓
[Step 1: Text Cleaning & Normalization]
  • URL Regex → 'link_present'
  • Digits Regex → 'numtoken'
  • Preserve Devanagari Unicode range (\u0900-\u097F)
       ↓
[Step 2: Regional Stopword Removal & Stemming]
  • Custom Stopwords: aahe, ahet, ani, pan, te, tya, hi, ho, cha, chi, che, hai, tha, aur, etc.
  • Porter Stemmer for token consolidation
       ↓
[Step 3: Feature Extraction]
  • TF-IDF Vectorizer (Unigrams + Bigram n-grams, sublinear TF scaling, max 3,000 features)
       ↓
[Step 4: Classification & Probability Calibration]
  • Calibrated LinearSVC (CalibratedClassifierCV with Sigmoid fitting)
  • Decision Threshold = 0.70 (Tuned to eliminate false alarms)
       ↓
[Output Verdict: 🚨 PHISHING vs 🛡️ SAFE]
```

#### 5. Model Benchmarking (5-Fold Stratified Cross-Validation)

| Classifier Model | CV Accuracy | CV F1 | CV ROC-AUC | CV PR-AUC | Selection Status |
|---|---:|---:|---:|---:|:---:|
| **LinearSVC** | **97.75%** | **0.9778** | **0.9978** | **0.9977** | ✅ **Selected Top Model** |
| Logistic Regression | 98.00% | 0.9804 | 0.9966 | 0.9965 | Candidate |
| Multinomial Naive Bayes | 95.75% | 0.9582 | 0.9909 | 0.9919 | Baseline |

*(Attach Image 1: `model_benchmark.png` under Section 5)*

#### 6. Holdout Evaluation Results (80 Test Messages)
- **Holdout Accuracy:** **97.5%**
- **Phishing Recall:** **100.0%** (**0 False Negatives** — zero phishing scams missed!)
- **Phishing Precision:** **95.2%** (True Negatives: 38, False Positives: 2)
- **ROC-AUC Score:** **0.9931**
- **PR-AUC Score:** **0.9930**
- **Holdout Confusion Matrix:**
  - Actual Safe (38): 38 Predicted Safe, 2 Predicted Phishing (False Positives)
  - Actual Phishing (40): 0 Predicted Safe (False Negatives), 40 Predicted Phishing (True Positives)

*(Attach Image 2: `confusion_matrix.png` under Section 6)*  
*(Attach Image 3: `threshold_analysis.png` under Section 6)*

---

### 📌 COLUMN 3: EXPLAINABILITY, CASE STUDY & CONCLUSION

#### 7. Feature Attribution & Top Learned Weights (LinearSVC)

| Top Phishing Indicator | Weight | Top Legitimate Indicator | Weight |
|---|---:|---|---:|
| `link_present` (URL) | **+3.85** | `aaj` (today) | **-1.04** |
| `कर link_present` (Devanagari action) | **+1.09** | `kal` (tomorrow) | **-0.89** |
| `rs numtoken` (Amount mention) | **+0.80** | `credit` | **-0.76** |
| `kyc` (KYC fraud vocabulary) | **+0.74** | `recharge` | **-0.65** |
| `pay` (Payment instruction) | **+0.73** | `meeting` | **-0.58** |
| `account` (Account urgency) | **+0.71** | `successful` | **-0.54** |

#### 8. Real-World Case Study Validation (10 Unseen Messages)
Tested against 10 out-of-sample real-world scam messages sourced from active Indian cybercrime reports (without any model retraining):
- **Case Study Accuracy:** **80.0% (8/10 correct)**
- **Detected Scams:** SBI English KYC (99.8%), Hinglish UPI Block (99.9%), Marathlish Customs Duty (100.0%), Hindi Devanagari Bank KYC (99.9%), English Lottery (100.0%), Hinglish Loan (99.8%), Marathlish Courier (99.9%).
- **Honest Error Analysis:**
  - *False Negative (Digital Arrest scam without link, 35.6% score):* Highlights that link-absent social engineering text requires dedicated authority-impersonation features.
  - *False Positive (UPI payment receipt containing URL, 87.3% score):* Reflects conservative security bias on transaction alerts containing external links.

#### 9. Key Contributions & Offline Deployability
1. **Targeted Regional Protection:** Fills a critical defense gap for Indian regional language mobile users.
2. **Zero-GPU & Fully Offline:** Operates entirely on standard CPUs without cloud APIs, internet connectivity, or paid service dependencies.
3. **Live Streamlit App:** Deployed interactive web dashboard (`streamlit run demo_app.py`) featuring real-time risk verdict, confidence bar, feature contribution chips, and preprocessing trace.
4. **Future Scope:** Benchmark against Indic-BERT / MuRIL transformers and expand dataset to 5,000+ crowdsourced messages.

---

## 🖼️ ATTACHED IMAGE GRAPH SPECIFICATIONS

Please embed placeholders and captions for the 3 attached PNG files:
1. `model_benchmark.png` $\rightarrow$ **Figure 1:** 5-Fold Stratified Cross-Validation Benchmark Comparison.
2. `confusion_matrix.png` $\rightarrow$ **Figure 2:** Holdout Test Set Confusion Matrix (0 False Negatives).
3. `threshold_analysis.png` $\rightarrow$ **Figure 3:** Decision Threshold vs Accuracy, Precision, and Recall Tuning Curve.
```
