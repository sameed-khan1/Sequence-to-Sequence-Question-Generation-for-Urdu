# Research Dossier: Sequence-to-Sequence Question Generation for Urdu From Scratch

## Authors & Metadata
- Project: Sequence-to-Sequence Question Generation for Urdu
- Authors: Sameed Khan & Project Partner
- Course: Generative AI (Assignment No. 01)
- Repository: https://github.com/sameed-khan1/Sequence-to-Sequence-Question-Generation-for-Urdu
- Model Architecture: 2-Layer Bidirectional LSTM Encoder + 2-Layer LSTM Decoder with Bahdanau Additive Attention
- Parameter Count: 27,102,016
- Target Task: Natural Urdu Question Generation from marked answer spans (<ans> ... </ans>)

---

## 1. Abstract & Motivation
Automatic Question Generation (QG) models human educator behavior by creating answerable, syntactically coherent questions from source text and target answer spans. While Question Answering (QA) tests comprehension, QG tests creative language generation. In Urdu, a low-resource Indo-Aryan language with complex right-to-left morphology and agglutinative syntax, QG has remained largely unexplored without relying on massive pretrained multilingial transformers (such as mT5 or mBART).

This project demonstrates that a 27.1-million parameter recurrent neural network built entirely from scratch using PyTorch primitives (nn.LSTM, nn.Linear, nn.Embedding) and trained on 75,067 filtered Urdu sentence pairs can learn to reliably generate accurate, entity-aware questions, achieving a validation perplexity of 74.69, a BLEU-4 score of 5.19 on out-of-domain Wikipedia text, and strong human evaluation agreement (Cohen's Kappa = 0.70).

---

## 2. Dataset & Preprocessing Pipeline
1. Source Dataset: UQA (Corpus for Urdu Question Answering), derived from SQuAD 2.0 translated into Urdu.
2. Filtering Steps:
   - Removed unanswerable rows.
   - Segmented paragraphs into sentences using Urdu punctuation markers (U+06D4 Urdu full stop, U+061F Urdu question mark, exclamation mark).
   - Wrapped the target answer span with `<ans>` and `</ans>`.
   - Discarded pairs where source > 60 words or target > 25 words to prevent recurrent vanishing gradients.
3. Dataset Splits:
   - Training Set: 75,067 sentence-question pairs (Mean source: 32.59 words, Mean target: 11.92 words).
   - Validation Set: 10,018 sentence-question pairs (Mean source: 33.23 words, Mean target: 12.29 words).
   - Out-of-Domain (OOD) Test: 177 Wiki-UQA pairs extracted from organic Urdu Wikipedia articles.

---

## 3. Subword Tokenizer: SentencePiece Unigram
- Tokenizer: SentencePiece Unigram algorithm.
- Vocabulary Size: 8,000 subwords.
- Special Tokens: `<pad>=0`, `<unk>=1`, `<s>=2`, `</s>=3`, `<ans>=4`, `</ans>=5`.
- Morphological Decomposition:
  - Urdu verbal inflections and postpositions are broken into linguistically sound stems and affixes (e.g., `پڑھا` -> `_پڑھ` + `_ا`, `فرانسیسیوں` -> `_فرانس` + `_یسی` + `_وں`).
  - Validation <unk> rate achieved: 0.10%, virtually eliminating out-of-vocabulary artifacts.

---

## 4. Deep Neural Architecture (27,102,016 Trainable Parameters)
- Encoder:
  - 2-layer Bidirectional LSTM, embedding size 256, hidden dimension 512 per direction (1024-dimensional combined output).
  - Pack padded sequences (pack_padded_sequence) applied to prevent the backward LSTM from traversing meaningless zero-padding tokens.
- Hidden State Bridging:
  - Four distinct linear projections bridge bidirectional encoder states to the decoder:
    - bridge_h0: [h_f0; h_b0] -> decoder_h0
    - bridge_h1: [h_f1; h_b1] -> decoder_h1
    - bridge_c0: [c_f0; c_b0] -> decoder_c0
    - bridge_c1: [c_f1; c_b1] -> decoder_c1
  - Prevents the catastrophic cross-layer summation bug common in naive Seq2Seq implementations.
- Decoder:
  - 2-layer unidirectional LSTM, hidden dimension 512, dropout 0.3.
  - Bahdanau Additive Attention: alignment score e_ti = v^T tanh(W_e h_i + W_d s_t).
  - Explicit <pad> masking with -infinity before softmax prevents attention leakage to padding tokens.
  - Input feeding: Concatenates previous context vector c_{t-1} with token embedding at each step.
  - Context fusion layer combines decoder hidden state s_t with context vector c_t before vocabulary projection.

---

## 5. Decoding Strategies
1. Greedy Search: Selects argmax token at each step. Fast, but susceptible to repetitive postposition loops.
2. Beam Search: Width k=3, length penalty alpha=0.7. Evaluates top candidate hypotheses globally.
3. Trigram Repetition Blocking: Dynamically sets logits to -1e9 if a candidate token would complete an identical trigram, completely eliminating stuttering loops (e.g., `کے ساتھ ساتھ`).

---

## 6. Training Dynamics & Quantitative Results
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU.
- Optimizer: Adam (lr=0.001), ReduceLROnPlateau (factor=0.5, patience=1), Gradient Clipping (max_norm=1.0).
- Epochs: 10 full epochs (batch size 64, ~6 minutes per epoch).
- Progression:
  - Epoch 1: Train Loss 5.8160 | Valid Loss 5.4093 | PPL 223.49
  - Epoch 5: Train Loss 4.2787 | Valid Loss 4.5607 | PPL 95.65
  - Epoch 10: Train Loss 3.6962 | Valid Loss 4.3134 | PPL 74.69
- Evaluation Benchmarks:
  - UQA Validation Greedy: BLEU-4 = 4.26, ROUGE-L = 0.2510, <unk> = 0.10%
  - UQA Validation Beam (k=3): BLEU-4 = 4.14, ROUGE-L = 0.2539, <unk> = 0.28%
  - Wiki-UQA (OOD) Greedy: BLEU-4 = 4.85, ROUGE-L = 0.2396, <unk> = 0.43%
  - Wiki-UQA (OOD) Beam (k=3): BLEU-4 = 5.19, ROUGE-L = 0.2467, <unk> = 0.07%
- Human Evaluation (50 Samples, 2 Members):
  - Fluency: Member 1 = 100%, Member 2 = 62%
  - Relevance: Member 1 = 92%, Member 2 = 86%, Cohen's Kappa = 0.70 (Strong agreement)
  - Answerability: Member 1 = 96%, Member 2 = 86%, Cohen's Kappa = 0.41 (Moderate agreement)

---

## 7. Qualitative Insights & Error Taxonomy
1. Best Interrogatives:
   - `کب` (When): Extremely high precision for dates, years (911, 1082), centuries.
   - `کون` / `کس نے` (Who): Reliable on leaders (Rollo, William the Conqueror) and titles.
   - `کس ملک` / `کہاں` (Where / Which country): Accurate on geographic entities (France, Germany).
2. Failure Types:
   - Particle stuttering (repetitive genitive `کے کے`).
   - Subword duplication (`ثقافتی ثقافتی`).
   - English loanword transliteration fragments (`آف`).
   - Entity hallucination (e.g., substituting World War I for Crusades).
3. Out-of-Domain Translationese:
   - UQA training data is translated from English SQuAD and exhibits rigid syntactic structures.
   - Wiki-UQA is organic human-written Wikipedia text. The model generalizes well (5.19 BLEU-4) but shows lexical shifts.

---

## 8. Shipped Front End & Verification
- Interactive Streamlit application (`app.py`).
- Instant toggling between Greedy and Beam search.
- Interactive visualization of Bahdanau Attention Heatmaps.
- Model checkpoint loaded in ~1.5 seconds.
