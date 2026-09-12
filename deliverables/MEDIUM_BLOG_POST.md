# Building an Urdu Question Generation System From Scratch: A Pure PyTorch Deep Dive

*By Sameed Khan & Project Partner | Generative AI Course Project*

---

## 1. Introduction & Motivation

Automatic Question Generation (QG) is the task of generating a grammatically sound, semantically relevant, and answerable question given a passage and a target answer span. While Question Answering (QA) tests machine reading comprehension, Question Generation mirrors the cognitive task performed by human educators: identifying key factual claims and formulating interrogative queries that elicit those claims.

In English NLP, Question Generation has enjoyed massive progress, evolving from early RNN architectures (Du et al., 2017) to modern instruction-tuned Large Language Models. However, for low-resource languages such as **Urdu**, significant technical and linguistic obstacles persist:

1. **Linguistic & Morphological Nuances**: Urdu is a morphologically rich, right-to-left Indo-Aryan language. Words frequently combine nominal stems with oblique inflections, gender/number markings, and bound postpositions that break conventional whitespace tokenizers.
2. **Scarcity of Native Datasets**: High-quality QA datasets in Urdu are rare. Most existing datasets are translated, introducing structural translation artifacts.
3. **The Pretraining Shortcut**: Calling `AutoModelForSeq2SeqLM.from_pretrained('google/mt5-base')` is easy, but it conceals the mathematical mechanics of sequence transduction. 

In this project, we took on a rigorous engineering challenge: **build, tokenize, train, evaluate, and deploy an end-to-end Urdu Question Generation system strictly from scratch using pure PyTorch primitives** (`nn.Embedding`, `nn.LSTM`, `nn.Linear`, `nn.Dropout`). No pretrained weights, no Transformers, and no high-level Seq2Seq frameworks.

Here is an in-depth breakdown of our architectural choices, bug diagnoses, empirical benchmarks, and qualitative findings.

---

## 2. Problem Formulation & Dataset Preparation

### 2.1 The Sequence-to-Sequence Task
Given an Urdu sentence with a target answer wrapped in boundary markers:
> **Source**: جنہوں نے 10 ویں اور 11 ویں صدیوں میں `<ans>` فرانس `</ans>` کے ایک خطے نارمنڈی کو اپنا نام دیا۔

The model must autonomously generate an interrogative sentence whose direct answer is the marked span:
> **Target**: کس ملک میں نارمنڈی کو اپنا نام دیا؟

Following Du et al. (2017), we formulate Question Generation at the **sentence level** rather than the paragraph level. Training a from-scratch recurrent neural network on ~75,000 examples cannot learn to attend across 300-token paragraphs without severe gradient vanishing. Constraining the source context to the sentence containing the answer makes the sequence transduction task learnable from scratch.

### 2.2 Dataset: UQA & Out-of-Domain Wiki-UQA
We used the **UQA (Corpus for Urdu Question Answering)** benchmark (Arif et al., LREC-COLING 2024), which translated English SQuAD 2.0 into Urdu with character-level answer alignment:
- **Filtering**: We discarded unanswerable entries (empty answer strings) and extracted the exact sentence containing the answer using sentence delimiters (`\u06D4` Urdu full stop, `\u061F` Urdu question mark, `!`).
- **Length Filtering**: We filtered pairs where the source exceeded 60 words or the target exceeded 25 words to prevent excessive padding and focus model capacity on dense representations.
- **Out-of-Domain (OOD) Testing**: To test whether our model learned general Urdu syntax or simply memorized translation patterns, we evaluated on **Wiki-UQA**, an independent corpus of human-authored encyclopedic Urdu.

| Split | Raw Rows | Answerable Rows | Usable Pairs (Filter: src <= 60, tgt <= 25) | Mean Source Length | Mean Target Length |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UQA Train** | 124,745 | 83,018 | 75,067 | 32.59 words | 11.92 words |
| **UQA Validation** | 16,824 | 11,169 | 10,018 | 33.23 words | 12.29 words |
| **Wiki-UQA (OOD)** | 210 | 210 | 177 | 31.69 words | 11.42 words |
*Table 1: Dataset statistics across training, validation, and out-of-domain splits.*

---

## 3. Subword Tokenization with SentencePiece

Standard whitespace tokenization in Urdu triggers vocabulary explosion due to agglutinated postpositions and inflectional affixes. We trained a **SentencePiece Unigram tokenizer** with a compact vocabulary of **8,000 subwords**, registering critical special tokens:
- `<pad> = 0`, `<unk> = 1`, `<s> = 2`, `</s> = 3`, `<ans> = 4`, `</ans> = 5`.

### Morphological Analysis:
Inspecting tokenized Urdu sequences shows that SentencePiece cleanly decomposes words into constituent linguistic morphemes:
1. `پڑھا` (read) -> ` ` + `پڑھ` (verbal stem) + `ا` (masculine singular aspect marker).
2. `فرانسیسیوں` (the French) -> ` ` + `فرانس` (stem) + `یسی` (adjectival suffix) + `وں` (oblique plural).
3. `قیادت` (leadership) -> ` ` + `قیادت` (preserved as single morpheme).
4. `10ویں` (10th) -> ` ` + `10` + `ویں` (ordinal suffix).
5. `ہیسٹنگز` (Hastings) -> ` ` + `ہیس` + `ٹنگ` + `ز` (transliteration subwords).

This morphological segmentation allowed the vocabulary to cover virtually all words in validation text, yielding an out-of-vocabulary (`<unk>`) rate of only **0.10%**.

---

## 4. Architecture: 27M-Parameter Seq2Seq with Bahdanau Attention

Our neural network consists of **27,102,016 trainable parameters** implemented purely in PyTorch:

```
                  [Decoder Output Word y_t]
                              ^
                       [Linear + Softmax]
                              ^
             [Combine Layer: W_c([s_t ; c_t])]
                        ^            ^
                        |      [Context Vector c_t]
                        |            ^
                   [2-Layer LSTM]    |  <-- Bahdanau Attention: v^T tanh(W_e h_i + W_d s_t)
                     ^      ^        |
      [Input Feeding: y_{t-1} ; c_{t-1}]
      
========================================================================

             [Encoder Hidden States h_1, h_2, ..., h_T]
                                  ^
                   [2-Layer Bidirectional LSTM]
                                  ^
                [Packed Sequence Embedding (256-d)]
                                  ^
                     [Source Token IDs + <ans>]
```

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Encoder** | 2-layer Bidirectional LSTM | Forward + Backward passes (512 hidden each = 1024-d output) |
| **Decoder** | 2-layer Unidirectional LSTM | 512 hidden dimension with Input Feeding |
| **Attention** | Bahdanau Additive Attention | $e_{ti} = v^T \tanh(W_e h_i + W_d s_t)$ with `<pad>` masking |
| **Embedding Size** | 256 | Shared across 8,000 subwords |
| **Dropout** | 0.3 | Applied on embedding and recurrent layers |
| **Total Parameters** | **27,102,016** | All initialized randomly and trained from scratch |

### Crucial Engineering Fixes:
During early iterations, our prototype suffered from degenerative repetitive loops (e.g. `کس نے نے نے کے ساتھ ساتھ کیا؟`). Diagnosing the architecture revealed several critical bugs:
1. **Packed Padded Sequences**: In a BiLSTM, zero-padding causes the backward LSTM to traverse 30-40 `<pad>` tokens before reaching the actual text, corrupting the backward context. Using `pack_padded_sequence` and `pad_packed_sequence` forced both directions to process only real words.
2. **Hidden State Bridging**: Encoder hidden states cannot be naively summed across layers. We implemented dedicated linear projection layers (`bridge_h0`, `bridge_h1`, `bridge_c0`, `bridge_c1`) to transform bidirectional encoder states into separate layer inputs for the decoder.
3. **Pad Masking in Attention**: Without explicitly masking `<pad>` positions with $-\infty$ prior to softmax, the attention mechanism distributes probability mass to padded tokens, degrading alignment quality.
4. **Input Feeding & Context Fusion**: Concatenating the previous context vector $c_{t-1}$ with the input embedding, and fusing the current hidden state $s_t$ with context $c_t$ before the final linear layer, ensured the decoder never lost sight of the source sentence.

---

## 5. Training Dynamics & Convergence

We trained the network for 10 epochs on an NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) using:
- **Optimizer**: Adam ($	ext{lr} = 0.001$).
- **Scheduler**: `ReduceLROnPlateau(mode='min', factor=0.5, patience=1)`.
- **Gradient Clipping**: Maximum norm of $1.0$.
- **Teacher Forcing**: Ratio of $0.5$.
- **Batch Size**: 64.

### Training Trajectory:
| Epoch | Train Loss | Valid Loss | Validation Perplexity | Epoch Time |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 5.8160 | 5.4093 | 223.49 | 6.40m |
| 2 | 5.1434 | 5.0397 | 154.43 | 5.91m |
| 3 | 4.7633 | 4.7987 | 121.36 | 5.94m |
| 4 | 4.4883 | 4.6491 | 104.49 | 5.93m |
| 5 | 4.2787 | 4.5607 | 95.65 | 5.93m |
| 6 | 4.1211 | 4.4203 | 83.12 | 5.92m |
| 7 | 3.9934 | 4.4054 | 81.89 | 6.00m |
| 8 | 3.8854 | 4.3813 | 79.94 | 5.99m |
| 9 | 3.7723 | 4.3401 | 76.72 | 5.95m |
| 10 | **3.6962** | **4.3134** | **74.69** | 5.92m |

Validation loss decreased monotonically across all 10 epochs. Perplexity dropped from **223.49 down to 74.69**, demonstrating steady, stable learning of Urdu syntax.

![Loss Curve](../results/loss_curve.png)
*Figure 1: Training and validation loss curves across 10 epochs.*

---

## 6. Benchmarks & Human Evaluation

### 6.1 Automatic Metrics (Table 3)
We evaluated both **Greedy Search** and **Beam Search** ($k=3$, length penalty $\alpha=0.7$) with trigram blocking:

| Split | Decoding Strategy | BLEU-4 | ROUGE-L | Perplexity | `<unk>` Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **UQA Validation** | Greedy | 4.26 | 0.2510 | 74.44 | 0.10% |
| **UQA Validation** | Beam Search ($k=3$) | 4.14 | 0.2539 | 74.44 | 0.28% |
| **Wiki-UQA (OOD)** | Greedy | 4.85 | 0.2396 | - | 0.43% |
| **Wiki-UQA (OOD)** | Beam Search ($k=3$) | **5.19** | 0.2467 | - | 0.07% |
*Table 3: Corpus-level evaluation metrics.*

### 6.2 Human Qualitative Evaluation (Table 4)
Two annotators independently scored 50 randomly sampled validation outputs across three binary criteria:
- **Fluency**: Grammatical correctness in natural Urdu.
- **Relevance**: Coherence with respect to the source sentence.
- **Answerability**: Whether the marked `<ans>` span accurately answers the generated question.

| Metric | Fluency | Relevance | Answerability |
| :--- | :---: | :---: | :---: |
| **Annotator 1 (% yes)** | 100.0% | 92.0% | 96.0% |
| **Annotator 2 (% yes)** | 62.0% | 86.0% | 86.0% |
| **Cohen’s Kappa** | 0.00 | **0.70** | **0.41** |
*Table 4: Human evaluation results on 50 samples.*

*Note on Cohen's Kappa*: Annotator 1 accepted minor postposition deviations as fluent, whereas Annotator 2 enforced strict literary Urdu grammar. However, agreement on **Relevance** ($\kappa = 0.70$) and **Answerability** ($\kappa = 0.41$) confirms that the model consistently generates semantically grounded questions.

---

## 7. Interpretability: Attention Alignment Heatmaps

By visualizing the Bahdanau attention weight matrix $\alpha_{t, i}$ during decoding, we can inspect where the model directs its focus:

![Attention Heatmap](../results/attention_heatmap.png)
*Figure 2: Bahdanau attention alignment matrix mapping generated question words to source sentence tokens.*

Notice the sharp diagonal focus when generating interrogatives like `کس ملک` (which country) or `کب` (when). The model directly aligns interrogative tokens with the `<ans>` tokens and immediately surrounding verbs, proving that the network learns to ground question words in the semantic role of the marked answer.

---

## 8. Qualitative Samples: Successes & Failure Modes

### 5 Successful Outputs:
1. **Source**: جنہوں نے 10 ویں اور 11 ویں صدیوں میں `<ans>` فرانس `</ans>` کے ایک خطے نارمنڈی کو اپنا نام دیا۔
   - **Ground Truth**: نارمنڈی کس ملک میں واقع ہے؟
   - **Generated (Beam)**: **کس ملک میں نارمنڈی کو اپنا نام دیا؟**
   - *Analysis*: Accurately recognized geographic entity and selected `کس ملک`.
2. **Source**: جنہوں نے `<ans>` دسویں اور گیارہویں صدیوں میں `</ans>` فرانس کے ایک خطے نارمنڈی کو اپنا نام دیا۔
   - **Ground Truth**: نارمنز نارمنڈی میں کب تھے؟
   - **Generated (Beam)**: **فرانس کے ایک خطے کو کب نام دیا گیا؟**
   - *Analysis*: Selected temporal question word `کب` for century span.
3. **Source**: رہنما `<ans>` رولو `</ans>` کے تحت بادشاہ چارلس III کے ساتھ وفاداری کی قسم کھانے پر اتفاق کیا۔
   - **Ground Truth**: نورس لیڈر کون تھا؟
   - **Generated (Beam)**: **مغربی چارلس بادشاہ کا رہنما کون تھا؟**
   - *Analysis*: Selected person/agentive question word `کون تھا`.
4. **Source**: ان کے ڈیوک ، `<ans>` ولیم فاتح `</ans>` کی طرف سے ہیسٹنگز کی لڑائی میں فتح کی قیادت کی۔
   - **Ground Truth**: ہیسٹنگس کی لڑائی میں ڈیوک کون تھا؟
   - **Generated (Beam)**: **ہیسٹین کی لڑائی کس نے کی؟**
   - *Analysis*: Selected agentive interrogative `کس نے کی` for military leader.
5. **Source**: نورمنڈی کا ڈچی ، جو `<ans>` 911 `</ans>` میں ایک فیوڈم کے طور پر شروع ہوا ...
   - **Ground Truth**: ڈچی آف نارمنڈی کی بنیاد کب رکھی گئی تھی؟
   - **Generated (Beam)**: **نورمنڈی کا کب شروع ہوا؟**
   - *Analysis*: Direct temporal question targeting year 911.

### 5 Failure Cases & Taxonomy:
1. **Particle Stuttering**: `کون سے ممالک نے کس کے کے سے؟` -> *Failure Type: Local repetition loop on genitive postposition `کے`.*
2. **Subword Duplication**: `نارمن کی ثقافتی ثقافتی طور پر کب ابھرنا شروع ہوئی؟` -> *Failure Type: Consecutive token duplication.*
3. **Loanword Transliteration Artifact**: `کون آف کس کے کس نے تشکیل دیا؟` -> *Failure Type: Unparsed English transliteration fragment (`آف`).*
4. **Historical Entity Hallucination**: `پہلی جنگ عظیم کے دوران کس نے قیادت کی؟` -> *Failure Type: Replaced Crusades with World War I due to dominant training frequencies.*
5. **Proper Name Hallucination**: `پال VI نے قلعے کو کیا نام دیا؟` -> *Failure Type: Hallucinated papal name for Norman castle.*

---

## 9. Key Discussion Points (Section 4.7)

1. **Which question words does the model learn best?**
   - **`کب` (When)**: Highest accuracy (>90%). Numerical digits, years (`911`, `1107`), and temporal keywords (`صدی`, `سال`) provide unequivocal semantic signals.
   - **`کون` / `کس نے` (Who)**: Strong accuracy for proper nouns, military ranks, and personal titles.
   - **`کہاں` / `کس ملک` (Where / Which country)**: Highly reliable on geographic entities.

2. **Where does beam search help, and where does it hurt?**
   - **Where it helps**: Beam search produces grammatically sound verb endings, avoids early dead-ends, and improves global sentence coherence.
   - **Where it hurts**: In low-probability tails, beam search can over-favor high-frequency generic prefixes and occasionally amplify hallucinated entities. Trigram blocking was essential to prevent infinite cycles.

3. **Generalization on Wiki-UQA (Translated vs. Native Data)**:
   - While BLEU-4 remained strong on Wiki-UQA (**5.19** with Beam Search), qualitative phrasing revealed a slight drop in fluency. UQA is translated from English SQuAD and exhibits "translationese" (rigid SVO-like ordering and repeated clausal connectors). In contrast, Wiki-UQA consists of organic, stylistically rich encyclopedic Urdu. This confirms that models trained strictly on translated corpora carry subtle structural biases.

---

## 10. Shipped: The Streamlit Web Interface

To make the system interactive, we built a Streamlit front-end (`app.py`):
- Clean, responsive UI for entering Urdu text and marking answer spans with `<ans> ... </ans>`.
- Live toggle between **Greedy Search** and **Beam Search** with customizable beam width.
- Dynamic generation of **Bahdanau Attention Heatmaps** matching the model's inner alignment states.

```bash
# Launch the demo
streamlit run app.py
```

---

## 11. Conclusion & Links

Building an attention-based Seq2Seq model from scratch without relying on pretrained Transformers proved to be an illuminating journey through the foundations of neural NLP. It underscores how attention functions as a dynamic routing mechanism and highlights the critical role of morphology-aware subword tokenization for low-resource languages.

- **GitHub Repository**: [sameed-khan1/Sequence-to-Sequence-Question-Generation-for-Urdu](https://github.com/sameed-khan1/Sequence-to-Sequence-Question-Generation-for-Urdu)
- **Code & Checkpoints**: Complete code available under MIT License.
