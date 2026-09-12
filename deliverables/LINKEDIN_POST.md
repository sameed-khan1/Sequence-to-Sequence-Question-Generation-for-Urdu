🚀 Excited to share our latest Generative AI project: **Sequence-to-Sequence Question Generation for Urdu — Built and Trained Strictly From Scratch!** 🇵🇰🤖

In modern NLP, fine-tuning pretrained Transformers like mT5 or mBART by importing a Hugging Face checkpoint has become standard practice. But what happens when you build a sequence transduction system **entirely from primitives** (`nn.Embedding`, `nn.LSTM`, `nn.Linear`)?

My project partner and I tackled this engineering challenge: designing, tokenizing, training, and shipping an end-to-end Neural Question Generation model for Urdu without any pretrained weights or third-party Seq2Seq abstractions.

---

### 🧠 Core Architectural Highlights (27.1M Parameters):
🔹 **Linguistic Tokenization**: Trained a SentencePiece Unigram subword tokenizer (8k vocabulary) tailored for Urdu's agglutinative morphology, achieving an out-of-vocabulary rate of just **0.10%**.
🔹 **2-Layer Bidirectional LSTM Encoder**: Utilized packed padded sequence mechanics to eliminate padding noise in the backward recurrent pass.
🔹 **Hidden State Bridging**: Implemented four dedicated linear transformations to cleanly bridge bidirectional encoder states to the decoder without layer collapse.
🔹 **2-Layer LSTM Decoder with Bahdanau Additive Attention**: Implemented attention pad-masking and input feeding ($[y_{t-1} ; c_{t-1}]$) to maintain sharp focus on source context.
🔹 **Beam Search ($k=3, lpha=0.7$) with Trigram Blocking**: Successfully eradicated degenerative repetitive loops (`کے کے` / `ساتھ ساتھ`).

---

### 📊 Results & Benchmarks:
📈 **Convergence**: 10 epochs on an NVIDIA RTX 4050 GPU dropped validation perplexity from **223.49 ➔ 74.69** (Valid Loss: 4.31).
🎯 **Automatic Metrics**: Achieved **5.19 BLEU-4** and **0.25 ROUGE-L** on the out-of-domain **Wiki-UQA** benchmark.
👁️ **Interpretability**: Bahdanau attention heatmaps confirm sharp alignment between generated question words (like `کس ملک` for countries or `کب` for dates) and the target `<ans>` answer spans!
👥 **Human Evaluation (50 Samples)**: Evaluated for Fluency (100%), Relevance (92%), and Answerability (96%) with strong inter-annotator agreement ($\kappa = 0.70$).

---

### 💻 Shipped & Interactive:
We deployed the trained model behind an interactive **Streamlit web application** (`app.py`), enabling live question generation and dynamic attention heatmap inspection!

🔗 **GitHub Repository**: https://github.com/sameed-khan1/Sequence-to-Sequence-Question-Generation-for-Urdu
📝 **Detailed Medium Deep Dive**: [Link to your published Medium Article]

A huge shoutout to our course instructor and teaching team for setting a challenge that pushed us to understand deep learning from the ground up!

#DeepLearning #PyTorch #NLP #UrduNLP #MachineLearning #GenerativeAI #Seq2Seq #ArtificialIntelligence #Python #Research #Streamlit
