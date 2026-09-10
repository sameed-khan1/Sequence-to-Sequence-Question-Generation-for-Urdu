import os
import torch
import torch.nn as nn
import sentencepiece as spm
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Urdu Question Generation",
    page_icon="❓",
    layout="wide"
)

PAD, UNK, BOS, EOS = 0, 1, 2, 3
VOCAB_SIZE = 8000
EMB_DIM = 256
HIDDEN_DIM = 512
DROPOUT = 0.3
CHECKPOINT_PATH = "checkpoints/best_model.pt"
TOKENIZER_PATH = "ur_sp.model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Model Definitions
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout
        )
        self.bridge_h0 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_h1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c0 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c1 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        h0 = torch.tanh(self.bridge_h0(torch.cat([hidden[0], hidden[1]], dim=1)))
        h1 = torch.tanh(self.bridge_h1(torch.cat([hidden[2], hidden[3]], dim=1)))
        c0 = torch.tanh(self.bridge_c0(torch.cat([cell[0], cell[1]], dim=1)))
        c1 = torch.tanh(self.bridge_c1(torch.cat([cell[2], cell[3]], dim=1)))
        dec_h = torch.stack([h0, h1], dim=0)
        dec_c = torch.stack([c0, c1], dim=0)

        return outputs, dec_h, dec_c

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_enc = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys, mask):
        q = self.W_dec(query).unsqueeze(1)
        k = self.W_enc(keys)
        energy = self.v(torch.tanh(q + k)).squeeze(2)
        energy = energy.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(energy, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(
            emb_dim + hidden_dim * 2, hidden_dim, num_layers=2,
            batch_first=True, dropout=dropout
        )
        self.W_c = nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask, prev_context):
        embedded = self.dropout(self.embedding(input_token))
        lstm_in = torch.cat([embedded, prev_context.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.lstm(lstm_in, (hidden, cell))
        query = output.squeeze(1)
        context, attn = self.attention(query, encoder_outputs, mask)
        concat = torch.cat([query, context], dim=1)
        attentive_h = torch.tanh(self.W_c(concat))
        logits = self.fc_out(self.dropout(attentive_h))
        return logits, hidden, cell, context, attn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

# -------------------------------------------------------------
# Caching Model & Tokenizer
# -------------------------------------------------------------
@st.cache_resource
def load_resources():
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    encoder = Encoder(VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, dropout=DROPOUT)
    decoder = Decoder(VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, dropout=DROPOUT)
    model = Seq2Seq(encoder, decoder).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        cp = torch.load(CHECKPOINT_PATH, map_location=device)
        sd = cp["model_state_dict"]
        key_map = {
            "encoder.hidden1.weight": "encoder.bridge_h0.weight",
            "encoder.hidden1.bias": "encoder.bridge_h0.bias",
            "encoder.hidden2.weight": "encoder.bridge_h1.weight",
            "encoder.hidden2.bias": "encoder.bridge_h1.bias",
            "encoder.cell1.weight": "encoder.bridge_c0.weight",
            "encoder.cell1.bias": "encoder.bridge_c0.bias",
            "encoder.cell2.weight": "encoder.bridge_c1.weight",
            "encoder.cell2.bias": "encoder.bridge_c1.bias",
            "decoder.attention.encoder_layer.weight": "decoder.attention.W_enc.weight",
            "decoder.attention.decoder_layer.weight": "decoder.attention.W_dec.weight",
            "decoder.attention.attention_layer.weight": "decoder.attention.v.weight",
            "decoder.combine.weight": "decoder.W_c.weight",
            "decoder.combine.bias": "decoder.W_c.bias",
            "decoder.output_layer.weight": "decoder.fc_out.weight",
            "decoder.output_layer.bias": "decoder.fc_out.bias",
        }
        remapped_sd = {key_map.get(k, k): v for k, v in sd.items()}
        model.load_state_dict(remapped_sd)
        epoch = cp.get("epoch", "N/A")
        val_loss = cp.get("valid_loss", "N/A")
    else:
        epoch = "Not found"
        val_loss = "Not found"

    model.eval()
    return sp, model, epoch, val_loss

sp, model, best_epoch, best_val_loss = load_resources()

# -------------------------------------------------------------
# Inference Functions
# -------------------------------------------------------------
def run_greedy(source_text, max_length=25):
    src_ids = [BOS] + sp.encode(source_text, out_type=int) + [EOS]
    src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
    lens_t = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
    mask = (src_t != PAD)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_t, lens_t)
        prev_context = torch.zeros(1, HIDDEN_DIM * 2, device=device)
        curr = torch.tensor([[BOS]], dtype=torch.long, device=device)
        gen_tokens = []
        attentions = []

        for _ in range(max_length):
            logits, hidden, cell, prev_context, attn = model.decoder(
                curr, hidden, cell, encoder_outputs, mask, prev_context
            )
            # Trigram blocking
            if len(gen_tokens) >= 2:
                last_bg = (gen_tokens[-2], gen_tokens[-1])
                for i in range(len(gen_tokens) - 2):
                    if (gen_tokens[i], gen_tokens[i+1]) == last_bg:
                        logits[0, gen_tokens[i+2]] = -1e9

            next_tok = logits.argmax(1).item()
            if next_tok == EOS:
                break
            gen_tokens.append(next_tok)
            attentions.append(attn[0].cpu().numpy())
            curr = torch.tensor([[next_tok]], dtype=torch.long, device=device)

    question = sp.decode(gen_tokens)
    return question, src_ids, gen_tokens, attentions

def run_beam(source_text, beam_width=3, max_length=25, alpha=0.7):
    src_ids = [BOS] + sp.encode(source_text, out_type=int) + [EOS]
    src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
    lens_t = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
    mask = (src_t != PAD)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_t, lens_t)
        prev_context = torch.zeros(1, HIDDEN_DIM * 2, device=device)

        beams = [([BOS], 0.0, hidden, cell, prev_context, False)]
        completed = []

        for _ in range(max_length):
            all_candidates = []
            for tokens, score, h, c, p_ctx, is_done in beams:
                if is_done:
                    completed.append((tokens, score))
                    continue

                curr = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                logits, next_h, next_c, next_ctx, _ = model.decoder(
                    curr, h, c, encoder_outputs, mask, p_ctx
                )

                if len(tokens) >= 3:
                    last_bg = (tokens[-2], tokens[-1])
                    for i in range(1, len(tokens) - 2):
                        if (tokens[i], tokens[i+1]) == last_bg:
                            logits[0, tokens[i+2]] = -1e9

                log_probs = torch.log_softmax(logits[0], dim=-1)
                top_scores, top_indices = torch.topk(log_probs, beam_width)

                for k in range(beam_width):
                    cand_tok = top_indices[k].item()
                    cand_score = score + top_scores[k].item()
                    cand_tokens = tokens + [cand_tok]
                    if cand_tok == EOS:
                        completed.append((cand_tokens, cand_score))
                    else:
                        all_candidates.append((cand_tokens, cand_score, next_h, next_c, next_ctx, False))

            if not all_candidates:
                break

            all_candidates.sort(key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)
            beams = all_candidates[:beam_width]

            if len(completed) >= beam_width * 2:
                break

        if completed:
            completed.sort(key=lambda x: x[1] / ((len(x[0]) - 1) ** alpha), reverse=True)
            best_tokens = completed[0][0]
        else:
            best_tokens = beams[0][0]

        out_tokens = [t for t in best_tokens if t not in (BOS, EOS)]
        return sp.decode(out_tokens)

# -----------------------------------------------------------------------------
# Streamlit User Interface
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.urdu-text {
    direction: rtl;
    text-align: right;
    font-size: 22px;
    font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', 'Urdu Typesetting', Arial;
    line-height: 2.0;
}
.urdu-output {
    direction: rtl;
    text-align: right;
    font-size: 24px;
    font-weight: bold;
    color: #1a446c;
    background-color: #f0f7ff;
    padding: 15px;
    border-radius: 8px;
    border-right: 5px solid #2b5c8f;
    line-height: 2.2;
}
</style>
""", unsafe_allow_html=True)

st.title("🇵🇰 Sequence-to-Sequence Question Generation for Urdu")
st.markdown("**Built from scratch** with 2-layer Bidirectional LSTM Encoder + 2-layer Attentive LSTM Decoder (Bahdanau Attention) & SentencePiece Tokenizer.")

with st.sidebar:
    st.header("⚙️ Model Details")
    st.write(f"**Device:** `{device}`")
    st.write(f"**Architecture:** 2-Layer BiLSTM Encoder + 2-Layer Attentive Decoder")
    st.write(f"**Parameters:** ~27.1M")
    st.write(f"**Best Epoch:** `{best_epoch}`")
    st.write(f"**Validation Loss:** `{best_val_loss if isinstance(best_val_loss, str) else f'{best_val_loss:.4f}'}`")
    st.divider()
    beam_k = st.slider("Beam Width (k)", min_value=1, max_value=5, value=3)

st.subheader("1. Enter an Urdu Sentence & Mark the Answer")
st.write("Wrap the answer within `<ans>` and `</ans>`. For example:")
st.code("نارمن وہ لوگ تھے جنہوں نے 10 ویں اور 11 ویں صدیوں میں <ans> فرانس </ans> کے ایک خطے نارمنڈی کو اپنا نام دیا۔", language="text")

# Example selection
examples = {
    "Example 1: فرانس (France)": "نارمن وہ لوگ تھے جنہوں نے 10 ویں اور 11 ویں صدیوں میں <ans> فرانس </ans> کے ایک خطے نارمنڈی کو اپنا نام دیا۔",
    "Example 2: ولیم فاتح (William the Conqueror)": "نارمن مہم جوئیوں نے جنوبی اٹلی کو فتح کرنے کے بعد ان کے ڈیوک ، <ans> ولیم فاتح </ans> کی قیادت میں انگلینڈ کی نارمن فتح حاصل کی۔",
    "Example 3: 911": "نورمنڈی کا ڈچی ، جو <ans> 911 </ans> میں ایک فیوڈم کے طور پر شروع ہوا ، سینٹ کلیئر سور ایپٹ کے معاہدے کے ذریعہ قائم کیا گیا تھا۔",
    "Example 4: 30,000": "1081 میں اس نے 300 بحری جہازوں میں <ans> 30،000 </ans> مردوں کی ایک فوج کی قیادت کی جو البانیہ کے ساحل پر اتری۔"
}

chosen_ex = st.selectbox("Or choose a pre-loaded sample:", ["(Custom input)"] + list(examples.keys()))
default_input = examples[chosen_ex] if chosen_ex != "(Custom input)" else "نارمن وہ لوگ تھے جنہوں نے 10 ویں اور 11 ویں صدیوں میں <ans> فرانس </ans> کے ایک خطے نارمنڈی کو اپنا نام دیا۔"

user_input = st.text_area("Urdu Sentence with Answer Marked:", value=default_input, height=100)

col1, col2 = st.columns(2)
with col1:
    generate_btn = st.button("🚀 Generate Question", type="primary", use_container_width=True)

if generate_btn and user_input.strip():
    if "<ans>" not in user_input or "</ans>" not in user_input:
        st.error("⚠️ Please mark the target answer in the sentence using `<ans> ... </ans>` tags.")
    else:
        with st.spinner("Generating Urdu questions..."):
            greedy_output, src_ids, gen_tokens, attentions = run_greedy(user_input)
            beam_output = run_beam(user_input, beam_width=beam_k)

        st.subheader("2. Generated Questions")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🎯 Greedy Decoding")
            st.markdown(f'<div class="urdu-output">{greedy_output}</div>', unsafe_allow_html=True)

        with c2:
            st.markdown(f"### 🔍 Beam Search (k={beam_k})")
            st.markdown(f'<div class="urdu-output">{beam_output}</div>', unsafe_allow_html=True)

        # Attention visualization
        if attentions:
            st.subheader("3. Attention Alignment Heatmap")
            src_pieces = [sp.id_to_piece(tid) for tid in src_ids]
            tgt_pieces = [sp.id_to_piece(tid) for tid in gen_tokens]
            attn_mat = torch.tensor(attentions).numpy()

            fig, ax = plt.subplots(figsize=(max(8, len(src_pieces)*0.25), max(5, len(tgt_pieces)*0.35)))
            cax = ax.imshow(attn_mat, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(src_pieces)))
            ax.set_xticklabels(src_pieces, rotation=90, fontsize=8)
            ax.set_yticks(range(len(tgt_pieces)))
            ax.set_yticklabels(tgt_pieces, fontsize=9)
            ax.set_xlabel("Source Subwords")
            ax.set_ylabel("Generated Subwords")
            ax.set_title("Bahdanau Attention Alignment")
            fig.colorbar(cax, label="Attention Weight")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
