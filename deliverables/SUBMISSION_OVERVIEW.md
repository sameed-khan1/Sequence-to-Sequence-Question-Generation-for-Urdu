# Assignment Submission Dossier & Deliverables Index

**Course**: Generative AI (Fall 2026)  
**Assignment**: Assignment No. 01 — Sequence-to-Sequence Question Generation for Urdu  
**Authors**: Sameed Khan & Project Partner  
**Target Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 13.1)  

---

## 📋 Required Deliverables Checklist

| # | Deliverable | Status | Location / Link |
| :-: | :--- | :---: | :--- |
| **1** | **Complete Code File** | ✅ Complete | [main.py](../main.py), [app.py](../app.py), [requirements.txt](../requirements.txt) |
| **2** | **GitHub Repository Link** | ✅ Live & Pushed | [github.com/sameed-khan1/Sequence-to-Sequence-Question-Generation-for-Urdu](https://github.com/sameed-khan1/Sequence-to-Sequence-Question-Generation-for-Urdu) |
| **3** | **Medium Blog Post** | ✅ Ready to Publish | [deliverables/MEDIUM_BLOG_POST.md](MEDIUM_BLOG_POST.md) (2,412 words) |
| **4** | **LinkedIn Post Summary** | ✅ Formatted & Ready | [deliverables/LINKEDIN_POST.md](LINKEDIN_POST.md) |
| **5** | **Project Poster (NotebookLM)** | ✅ Ready | [deliverables/NOTEBOOKLM_POSTER_SOURCE.md](NOTEBOOKLM_POSTER_SOURCE.md)<br>[deliverables/PROJECT_POSTER_CONTENT.md](PROJECT_POSTER_CONTENT.md)<br>[deliverables/project_poster.html](project_poster.html) |

---

## 1. Complete Code Files
All project code has been written and formatted strictly from scratch without relying on pretrained models, Hugging Face Seq2Seq classes, or external Transformers:
- **`main.py`**: End-to-end pipeline containing SentencePiece loading, PyTorch Dataset & DataLoader with `collate_fn`, 2-layer BiLSTM Encoder with `pack_padded_sequence`, 4-layer state bridging projections, 2-layer LSTM Decoder with Bahdanau Additive Attention and Input Feeding, Adam optimizer with `ReduceLROnPlateau`, 10-epoch training loop, validation loss checkpointing, Greedy decoding, Beam Search with trigram blocking, and automatic metric calculation (BLEU-4, ROUGE-L, Perplexity, `<unk>` rate).
- **`app.py`**: Interactive Streamlit web user interface with dynamic sample loaders, decoding strategy selector, and high-resolution attention heatmap generation.
- **`requirements.txt`**: Minimal, reproducible dependencies (`torch`, `sentencepiece`, `sacrebleu`, `rouge-score`, `streamlit`, `matplotlib`, `datasets`).

---

## 2. GitHub Repository Link
- **Public URL**: **https://github.com/sameed-khan1/Sequence-to-Sequence-Question-Generation-for-Urdu**
- **Repository Health**:
  - Clean `main` branch with only necessary files.
  - Checkpoints (325 MB) and heavy virtual environments (`.venv/`) safely excluded via `.gitignore` to adhere strictly to GitHub's file size policies.
  - Includes high-res visualizations (`results/loss_curve.png`, `results/attention_heatmap.png`), evaluation tables, and complete technical `README.md`.

---

## 3. Medium Blog Post
- **Local File**: `deliverables/MEDIUM_BLOG_POST.md`
- **Length**: ~2,400 words (exceeds the 800–1,500 word requirement).
- **Key Sections Covered**:
  1. Problem Motivation & Urdu Linguistic Nuances.
  2. The "From Scratch" Philosophy vs. Pretrained Shortcuts.
  3. Sentence-level Formulation (Du et al., 2017).
  4. SentencePiece Subword Tokenization & Morphological Analysis.
  5. 27M-Parameter Neural Architecture & Critical Engineering Fixes (BiLSTM padding packing, state bridging, attention pad masking, input feeding).
  6. Training Loss Dynamics & Convergence (10 Epochs, PPL: 223.49 -> 74.69).
  7. Automatic Metrics (Table 3) & Blind Human Evaluation (Table 4, Cohen's Kappa = 0.70).
  8. Attention Interpretability & Alignment Heatmaps.
  9. Qualitative Analysis: 5 Good vs. 5 Bad Generations with Error Taxonomy.
  10. Viva Discussion Points: Question Word Strengths (`کب`, `کس نے`), Beam Search tradeoffs, and Wiki-UQA Generalization (Native Urdu vs. Translated SQuAD).
  11. Interactive Streamlit Web UI.

---

## 4. LinkedIn Post Summary
- **Local File**: `deliverables/LINKEDIN_POST.md`
- **Format**: High-impact, professional copy designed for LinkedIn engagement.
- **Includes**:
  - Technical hook: From-scratch Seq2Seq for low-resource Urdu.
  - Core metrics: 27.1M parameters, 10 epochs, PPL 74.69, BLEU-4 5.19 on Wiki-UQA.
  - Highlight of Bahdanau attention and interactive Streamlit UI.
  - Placeholders for Medium article URL and GitHub repository.
  - Industry-relevant hashtags.

---

## 5. Project Poster (Designed Using NotebookLM)
We have provided three complementary formats to fulfill this requirement:

1. **NotebookLM Source Document (`deliverables/NOTEBOOKLM_POSTER_SOURCE.md`)**:
   - Clean, structured research dossier formatted specifically for Google NotebookLM.
   - Upload this file to [notebooklm.google.com](https://notebooklm.google.com) to generate briefing documents, study guides, or an Audio Overview (AI podcast discussion).
2. **Poster Specification & Layout Guide (`deliverables/PROJECT_POSTER_CONTENT.md`)**:
   - Step-by-step block content for Canva, PowerPoint, or LaTeX poster templates.
3. **Interactive / Print-Ready HTML Poster (`deliverables/project_poster.html`)**:
   - Publication-quality academic conference poster styled with CSS Grid and modern typography.
   - Open in Google Chrome or Microsoft Edge and press `Ctrl + P` to save as a high-resolution PDF poster.
