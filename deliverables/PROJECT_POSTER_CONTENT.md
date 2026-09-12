# Scientific Research Poster: Content Specification & Layout Guide

> **How to use with Google NotebookLM**:
> 1. Go to [notebooklm.google.com](https://notebooklm.google.com).
> 2. Create a new notebook titled: **"Urdu Seq2Seq Question Generation"**.
> 3. Click **"Add Sources"** -> Upload `deliverables/NOTEBOOKLM_POSTER_SOURCE.md` (or copy-paste its text).
> 4. In the chat box, type:
>    - *"Generate an executive briefing document summarizing the core architecture, results, and failure modes."*
>    - *"Create a 5-bullet high-impact summary suitable for an academic conference poster header."*
>    - *"Generate an Audio Overview (Deep Dive podcast) discussing the project between two AI hosts."*
> 5. You can also view or print the ready-made, high-resolution HTML poster in `deliverables/project_poster.html` directly in your browser!

---

# Poster Layout Plan (A1 / A0 Landscape, 3-Column Standard)

```
+---------------------------------------------------------------------------------------------------------+
|                                           HEADER & TITLE BLOCK                                          |
|            SEQUENCE-TO-SEQUENCE QUESTION GENERATION FOR URDU (BUILT & TRAINED FROM SCRATCH)             |
|              Sameed Khan & Project Partner  |  Generative AI Lab  |  NVIDIA RTX 4050 GPU                |
|                    GitHub: github.com/sameed-khan1/Sequence-to-Sequence-Question-Generation-for-Urdu    |
+------------------------------------+------------------------------------+-------------------------------+
|         COLUMN 1: MOTIVATION       |        COLUMN 2: ARCHITECTURE      |       COLUMN 3: RESULTS       |
|            & TOKENIZATION          |            & TRAINING              |          & DISCUSSION         |
+------------------------------------+------------------------------------+-------------------------------+
| 1. Problem & Motivation            | 4. 27M-Param Neural Architecture   | 7. Quantitative Results       |
|    - Why Urdu Question Generation? |    - BiLSTM Encoder (2-layer)      |    - Table 1: Dataset Stats   |
|    - From-Scratch Constraint       |    - State Bridging (No Collapse)  |    - Table 3: BLEU & ROUGE    |
|    - Sentence-level Formulation    |    - Bahdanau Attention + Input Fed|    - Table 4: Human Eval      |
|                                    |                                    |                               |
| 2. Dataset Processing (UQA)        | 5. Training Dynamics               | 8. Attention Visualization    |
|    - 75,067 Train / 10,018 Valid   |    - 10 Epochs, Monotonic Loss     |    - Alignment Heatmap        |
|    - Wiki-UQA OOD Benchmark       |    - PPL: 223.49 -> 74.69          |    - Dynamic Answer Grounding |
|                                    |    - Loss Curve Graph              |                               |
| 3. SentencePiece Tokenizer         |                                    | 9. Qualitative Samples & Viva |
|    - 8,000 Unigram Subwords        | 6. Decoding Search Dynamics        |    - Good vs. Bad Examples    |
|    - Morphological Splitting       |    - Greedy vs. Beam (k=3)         |    - Error Taxonomy           |
|    - <unk> Rate: 0.10%             |    - Trigram Repetition Blocking   |    - Live Streamlit Web UI    |
+------------------------------------+------------------------------------+-------------------------------+
|                                            FOOTER & ACKNOWLEDGMENTS                                     |
+---------------------------------------------------------------------------------------------------------+
```

---

## Block-by-Block Poster Copy

### BLOCK 1: Title & Author Block
- **Title**: Sequence-to-Sequence Question Generation for Urdu
- **Subtitle**: Built, Tokenized, and Trained Strictly from Scratch Using PyTorch Primitives
- **Authors**: Sameed Khan & Team Partner
- **Affiliation**: Generative AI Course, Department of Computer Science
- **Code**: `github.com/sameed-khan1/Sequence-to-Sequence-Question-Generation-for-Urdu`

---

### BLOCK 2: Problem Formulation & Motivation
- **The Core Task**: Given an Urdu sentence with a designated target answer wrapped in `<ans> ... </ans>`, autonomously generate an answerable, grammatically coherent interrogative sentence.
- **Why From Scratch?** Pretrained transformers (mT5, mBART) hide sequence transduction mechanics. By implementing recurrent cells, attention projections, and beam decoding directly from primitives (`nn.Embedding`, `nn.LSTM`, `nn.Linear`), we examine the true inductive biases of Seq2Seq models.
- **Sentence-Level Constraint**: Following Du et al. (2017), isolating the answer-bearing sentence prevents gradient vanishing across 300-word paragraph contexts and makes training learnable on ~75k pairs.

---

### BLOCK 3: Dataset Pipeline (UQA + Wiki-UQA)
- **Corpus**: UQA (Urdu Question Answering) benchmark translated from SQuAD 2.0.
- **Preprocessing**: Answer start character alignment, Urdu sentence segmentation (`۔`, `؟`, `!`), length filtering (source $\le 60$, target $\le 25$).
- **Data Splits**:
  - Training: 75,067 pairs (mean src: 32.59 words, tgt: 11.92 words).
  - Validation: 10,018 pairs (mean src: 33.23 words, tgt: 12.29 words).
  - Out-of-Domain Test: 177 Wiki-UQA pairs from organic Urdu Wikipedia.

---

### BLOCK 4: Tokenizer & Urdu Morphology
- **SentencePiece Unigram Tokenizer**: 8,000 subwords.
- **Special Tokens**: `<pad>=0`, `<unk>=1`, `<s>=2`, `</s>=3`, `<ans>=4`, `</ans>=5`.
- **Morphological Splitting**:
  - `پڑھا` -> `_پڑھ` (read stem) + `_ا` (masculine aspect marker).
  - `فرانسیسیوں` -> `_فرانس` (stem) + `_یسی` (demonym) + `_وں` (oblique plural).
- **Out-of-Vocabulary (<unk>) Rate**: **0.10%** on validation text.

---

### BLOCK 5: Architecture (27,102,016 Trainable Parameters)
- **BiLSTM Encoder**: 2 layers, 256 embedding dimension, 512 hidden dimension. Uses `pack_padded_sequence` to prevent backward pass corruption on zero-padding.
- **Hidden State Bridging**: 4 distinct linear projections (`bridge_h0`, `bridge_h1`, `bridge_c0`, `bridge_c1`) mapping bidirectional states to decoder states without layer collapse.
- **Bahdanau Additive Attention**: Alignment score $e_{ti} = v^T 	anh(W_e h_i + W_d s_t)$ with explicit pad masking ($-\infty$).
- **Input Feeding**: Previous context vector $c_{t-1}$ concatenated with current token embedding before decoder recurrence.

---

### BLOCK 6: Training Dynamics & Convergence
- **Hardware**: NVIDIA GeForce RTX 4050 GPU (6GB VRAM).
- **Optimization**: Adam ($\eta = 0.001$), `ReduceLROnPlateau(factor=0.5, patience=1)`, gradient clipping ($1.0$).
- **Results**:
  - Epoch 1 Valid Loss: 5.4093 (PPL: 223.49)
  - Epoch 5 Valid Loss: 4.5607 (PPL: 95.65)
  - Epoch 10 Valid Loss: **4.3134** (PPL: **74.69**, Train Loss: **3.6962**)
- Continuous monotonic improvement without divergence.

---

### BLOCK 7: Benchmark Results & Human Evaluation
- **Automatic Metrics**:
  - UQA Validation Greedy: BLEU-4 = 4.26, ROUGE-L = 0.2510
  - UQA Validation Beam ($k=3$): BLEU-4 = 4.14, ROUGE-L = 0.2539
  - Wiki-UQA (OOD) Greedy: BLEU-4 = 4.85, ROUGE-L = 0.2396
  - Wiki-UQA (OOD) Beam ($k=3$): **BLEU-4 = 5.19**, ROUGE-L = 0.2467
- **Human Evaluation (50 Samples, 2 Annotators)**:
  - Fluency: 100% (Ann 1) / 62% (Ann 2)
  - Relevance: 92% (Ann 1) / 86% (Ann 2) — Cohen's $\kappa = 0.70$ (Strong Agreement)
  - Answerability: 96% (Ann 1) / 86% (Ann 2) — Cohen's $\kappa = 0.41$ (Moderate Agreement)

---

### BLOCK 8: Attention Heatmap & Interpretability
- Heatmap reveals sharp diagonal attention peaks over `<ans> ... </ans>` when generating interrogative pronouns (`کب`, `کس ملک`, `کون`).
- Validates that question syntax is directly conditioned on the semantic class of the marked entity.

---

### BLOCK 9: Qualitative Samples & Error Taxonomy
- **Successful Generations**:
  - Country entity: `کس ملک میں نارمنڈی کو اپنا نام دیا؟`
  - Temporal date: `فرانس کے ایک خطے کو کب نام دیا گیا؟`
  - Battle leader: `ہیسٹین کی لڑائی کس نے کی؟`
- **Failure Taxonomy**:
  1. Particle repetition loops (`کے کے`).
  2. Subword duplication (`ثقافتی ثقافتی`).
  3. Historical entity hallucination (substituting World War I for Crusades).
  4. English loanword artifacts (`آف`).

---

### BLOCK 10: Shipped Streamlit Web Demo
- End-to-end interactive demo built with Streamlit (`app.py`).
- Instant toggling between Greedy Search and Beam Search.
- Live attention heatmap generation.
