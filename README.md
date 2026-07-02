# Natural Language Inference (NLI): Fine-Tuned BERT vs. Zero-Shot Gemini LLM

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35%2B-F7D54E.svg)](https://huggingface.co/docs/transformers/index)
[![Gemini API](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-4285F4.svg)](https://ai.google.dev/)

An experimental Natural Language Processing (NLP) project exploring and comparing two distinct paradigms for **Natural Language Inference (NLI)** on the **Stanford Natural Language Inference (SNLI)** dataset:
1. **Task-Specific Fine-Tuning**: Fine-tuning a local encoder-only transformer (`bert-base-uncased`) for sequence classification.
2. **Zero-Shot LLM Prompting**: Leveraging a general-purpose Large Language Model (**Google Gemini 1.5 Flash**) via structured batch prompting without task-specific training.

---

## Comparative Performance Summary

Both models were evaluated on sentence pairs from the SNLI dataset to classify relationships into **Entailment**, **Neutral**, or **Contradiction**.

| Model Approach | Model Architecture | Accuracy | F1-Score | Invalid Rate | Latency & Execution | Cost |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Fine-Tuned Encoder** | `bert-base-uncased` | **85.30%** | **0.8519** | `0.00%` | Ultra-fast (~ms/batch), Local GPU/CPU | **Free** (post-training) |
| **Zero-Shot Generative LLM** | `gemini-1.5-flash` | **93.20%** | **0.9324** | `0.00%` | Network API dependent (~2-3s/batch) | Pay-per-prompt |

### Key Findings
* **Performance**: The zero-shot Gemini 1.5 Flash model achieved a remarkable **93.20% accuracy**, outperforming the fine-tuned BERT model (**85.30%**) significantly, demonstrating the advanced semantic reasoning capabilities of modern LLMs.
* **Reliability & Parsing**: Through carefully structured batch prompting requesting strictly formatted JSON list outputs, the Gemini API achieved a **0% formatting error / invalid rate** across 1,000 test samples.
* **Error Analysis**: Both models experienced the highest confusion around the **Neutral** class—a well-known challenge in NLI where distinguishing subtle nuances between logical entailment and neutral statements is difficult even for human annotators.
* **Production Recommendation (Hybrid Pipeline)**: For practical real-world applications (such as verification layers in **Retrieval-Augmented Generation / RAG** pipelines), a **Hybrid System** is recommended:
  1. Route sentence pairs first through the local **fine-tuned BERT** model for instant, zero-cost classification.
  2. If BERT outputs a high confidence score, accept the prediction immediately.
  3. If BERT's confidence is low, escalate the specific pair to **Gemini 1.5 Flash** as a high-accuracy fallback.

---

## Repository Structure

```text
├── 2220765019.ipynb     # Full experimental notebook (EDA, BERT fine-tuning, Gemini evaluation & discussion)
├── PA5_Colab.ipynb      # Streamlined notebook version optimized for Google Colab execution
├── requirements.txt     # Project Python dependencies
├── .gitignore           # Excludes heavy model checkpoints (>100MB) and temporary files
└── data/
    ├── train.parquet      # SNLI training dataset split
    ├── validation.parquet # SNLI validation dataset split
    └── test.parquet       # SNLI test dataset split
```

---

## Getting Started

### 1. Installation & Environment Setup
Clone the repository and install the required Python packages:

```bash
git clone https://github.com/your-username/your-nli-repo.git
cd your-nli-repo
pip install -r requirements.txt
```

### 2. Loading Pre-trained BERT Weights
By default, the notebooks utilize `bert-base-uncased`. When executing the code for the first time, the Hugging Face `transformers` library will automatically fetch and cache the base model weights (~440MB) directly from the Hugging Face Hub.

### 3. Configuring Google Gemini API
To execute Task 3 (Zero-Shot LLM Evaluation), you must obtain an API key from [Google AI Studio](https://aistudio.google.com/).
Inside the notebook, update the API key variable prior to running the Gemini inference cells:

```python
import google.generativeai as genai

API_KEY = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=API_KEY)
```

---

## Pipeline Methodology

1. **Exploratory Data Analysis & Preprocessing**:
   * Analyzed premise and hypothesis token length distributions, class balance, and missing/invalid labels (`-1` labels removed).
2. **BERT Fine-Tuning (Task 2)**:
   * Formatted sentence pairs as `[CLS] Premise [SEP] Hypothesis [SEP]`.
   * Trained using `AdamW` optimizer (`lr=2e-5`) with cross-entropy loss on CUDA GPU hardware.
3. **LLM Evaluation (Task 3)**:
   * Designed a batch-processing prompt sending 100 sentence pairs per API call to minimize network latency and token overhead.
   * Parsed raw JSON list responses cleanly into classification predictions.
4. **Comparative Visualization & Report (Tasks 4 & 5)**:
   * Generated side-by-side confusion matrices and classification reports using `seaborn` and `scikit-learn`.

---

## Academic Context
This project was developed as part of the **AIN442 - Natural Language Processing / Deep Learning** course coursework at Hacettepe University.
