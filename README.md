# Interactive-Text-Analysis-Tool
A Streamlit app for analyzing, improving and paraphrasing English text. It includes grammar/style checks, readability metrics, sentiment analysis, simple plagiarism checks (by embeddings), and paraphrasing using lightweight transformer models.
#  Interactive Text Analysis & Content Improvement Tool

[![Streamlit](https://img.shields.io/badge/streamlit-app-orange?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

>  A Streamlit app to analyze, improve and paraphrase English text: grammar/style checks, readability metrics, sentiment analysis, simple similarity-based plagiarism checks, and lightweight paraphrasing.


##  About
This repository contains a Streamlit app that inspects and improves English text. It runs several NLP tasks (grammar/style suggestions, tokenization, POS & lemmatization, readability scores, sentiment analysis, simplification, paraphrasing with a small transformer, and a basic similarity-based plagiarism check using sentence embeddings).

---

##  Features
- ✅ Grammar & style suggestions (`language_tool_python`)  
- ✅ Text cleaning (tokenization, stopwords)  
- ✅ POS tagging & lemmatization  
- ✅ Readability metrics (Flesch, Flesch-Kincaid, Gunning Fog, ...)  
- ✅ Sentiment analysis (TextBlob + VADER)  
- ✅ Sentence simplification & word suggestions  
- ✅ Paraphrasing via small seq2seq model (e.g., `t5-small`)  
- ✅ Basic plagiarism-like checks via `sentence-transformers` + cosine similarity

---

## 📁 Files
- `app.py` — main Streamlit app  
- `requirements.txt` — Python dependencies (add to repo)  
- (optional) `Dockerfile` — for container deployment  
- (optional) `Procfile` — for some platforms

---

## 🧰 Requirements
- Python 3.8+  
- pip
  
## How to run the Text Analysis Tool

A Streamlit app for analyzing text grammar, style, sentiment, and simple similarity-based plagiarism.

### Prerequisites
1. **Python 3.8+** installed.  
2. **Java 17+** installed **only if** you use the LanguageTool grammar checker. (If you disable LanguageTool, Java is not required.)

Verify Java (if needed):
```bash
java -version
# Should print something like: openjdk version "17.0.x"  or similar

**Recommended `requirements.txt`** (see file below):
