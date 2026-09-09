# Linguistic-Aware Phishing Detection for Regional Indian Languages

A lightweight, deployable phishing detection tool for Indian users who communicate in code-switched regional languages (Hinglish, Marathlish, Hindi/Marathi Devanagari).

## Quick Start

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy nltk matplotlib joblib streamlit
```

### 2. Train the model
```bash
python NLP.py --final-model LinearSVC --threshold 0.7
```
This generates `phishing_detector_model.joblib` and all evaluation artifacts.

### 3. Run the live demo
```bash
python -m streamlit run demo_app.py
```
Open **http://localhost:8501** in your browser.

### 4. Run the case study validation
```bash
python run_case_study.py
```

## Project Structure

| File | Purpose |
|---|---|
| `NLP.py` | Full training pipeline — preprocessing, benchmarking, evaluation |
| `demo_app.py` | Streamlit live demo app |
| `blind_test.py` | CLI blind-test for single messages |
| `run_case_study.py` | Case study validation script |
| `phishing_dataset_400_fixed.csv` | Raw labeled dataset (400 messages) |
| `case_study_messages.csv` | 10 out-of-sample validation messages |
| `poster_summary.md` | Poster-ready results summary |

## Important Notes

- The `.joblib` model file is **not included** in the repo (too large). Run `python NLP.py --final-model LinearSVC --threshold 0.7` to generate it locally.
- The demo runs fully **offline** — no API keys or internet needed.
