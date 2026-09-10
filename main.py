import os
import math
import time
import csv
import io
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sentencepiece as spm
import sacrebleu
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# UTF-8 Output
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding="utf-8",
    errors="replace"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Load Tokenizer
sp = spm.SentencePieceProcessor(
    model_file="ur_sp.model"
)

PAD = 0
UNK = 1
BOS = 2
EOS = 3

VOCAB_SIZE = 8000
EMBEDDING_SIZE = 256
HIDDEN_SIZE = 512
DROPOUT = 0.3

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# ---------------------------------------------------------------------------------------------------------

# Setup directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ---------------------------------------------------------------------------------------------------------

# Dataset And Collection
def load_pairs(file_path):

    pairs = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split("\t")

            if len(parts) == 2:
                pairs.append(
                    (parts[0], parts[1])
                )

    return pairs


train_pairs = load_pairs("train.tsv")
valid_pairs = load_pairs("valid.tsv")
wiki_pairs = load_pairs("wiki_test.tsv")


class UQADataset(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        source = self.pairs[index][0]
        target = self.pairs[index][1]

        source_ids = [BOS] + sp.encode(
            source,
            out_type=int
        ) + [EOS]

        target_ids = [BOS] + sp.encode(
            target,
            out_type=int
        ) + [EOS]

        return source_ids, target_ids


def collate_fn(batch):

    sources = []
    targets = []

    for source, target in batch:
        sources.append(source)
        targets.append(target)

    max_source_length = max(
        len(source)
        for source in sources
    )

    max_target_length = max(
        len(target)
        for target in targets
    )

    padded_sources = []
    padded_targets = []

    for source in sources:

        source = source + [PAD] * (
            max_source_length - len(source)
        )

        padded_sources.append(source)

    for target in targets:

        target = target + [PAD] * (
            max_target_length - len(target)
        )

        padded_targets.append(target)

    return (
        torch.tensor(
            padded_sources,
            dtype=torch.long
        ),
        torch.tensor(
            padded_targets,
            dtype=torch.long
        )
    )


train_loader = DataLoader(
    UQADataset(train_pairs),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    UQADataset(valid_pairs),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# ---------------------------------------------------------------------------------------------------------

# Model Architecture
class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, dropout=0.3):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=PAD
        )

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.hidden1 = nn.Linear(
            hidden_size * 2,
            hidden_size
        )

        self.hidden2 = nn.Linear(
            hidden_size * 2,
            hidden_size
        )

        self.cell1 = nn.Linear(
            hidden_size * 2,
            hidden_size
        )

        self.cell2 = nn.Linear(
            hidden_size * 2,
            hidden_size
        )

    def forward(self, source, lengths):

        embedded = self.embedding(source)
        embedded = self.dropout(embedded)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        hidden1 = torch.tanh(
            self.hidden1(
                torch.cat((hidden[0], hidden[1]), dim=1)
            )
        )

        hidden2 = torch.tanh(
            self.hidden2(
                torch.cat((hidden[2], hidden[3]), dim=1)
            )
        )

        cell1 = torch.tanh(
            self.cell1(
                torch.cat((cell[0], cell[1]), dim=1)
            )
        )

        cell2 = torch.tanh(
            self.cell2(
                torch.cat((cell[2], cell[3]), dim=1)
            )
        )

        hidden = torch.stack(
            (hidden1, hidden2),
            dim=0
        )

        cell = torch.stack(
            (cell1, cell2),
            dim=0
        )

        return output, hidden, cell


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.encoder_layer = nn.Linear(
            hidden_size * 2,
            hidden_size,
            bias=False
        )

        self.decoder_layer = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.attention_layer = nn.Linear(
            hidden_size,
            1,
            bias=False
        )

    def forward(self, hidden, encoder_output, mask):

        hidden = self.decoder_layer(hidden).unsqueeze(1)

        encoder_part = self.encoder_layer(
            encoder_output
        )

        energy = torch.tanh(
            hidden + encoder_part
        )

        scores = self.attention_layer(
            energy
        ).squeeze(2)

        scores = scores.masked_fill(
            ~mask,
            -1e9
        )

        attention = torch.softmax(
            scores,
            dim=1
        )

        context = torch.bmm(
            attention.unsqueeze(1),
            encoder_output
        )

        context = context.squeeze(1)

        return context, attention


class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, dropout=0.3):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=PAD
        )

        self.dropout = nn.Dropout(dropout)

        self.attention = BahdanauAttention(
            hidden_size
        )

        self.lstm = nn.LSTM(
            embedding_size + hidden_size * 2,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.combine = nn.Linear(
            hidden_size + hidden_size * 2,
            hidden_size
        )

        self.output_layer = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(
        self,
        input_token,
        hidden,
        cell,
        encoder_output,
        mask,
        previous_context
    ):

        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)

        lstm_input = torch.cat(
            (
                embedded,
                previous_context.unsqueeze(1)
            ),
            dim=2
        )

        output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        decoder_hidden = output.squeeze(1)

        context, attention = self.attention(
            decoder_hidden,
            encoder_output,
            mask
        )

        combined = torch.cat(
            (
                decoder_hidden,
                context
            ),
            dim=1
        )

        combined = torch.tanh(
            self.combine(combined)
        )

        prediction = self.output_layer(
            self.dropout(combined)
        )

        return (
            prediction,
            hidden,
            cell,
            context,
            attention
        )


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        source,
        source_lengths,
        target,
        teacher_forcing_ratio=0.5
    ):

        batch_size = source.size(0)
        target_length = target.size(1)
        vocab_size = self.decoder.output_layer.out_features

        mask = source != PAD

        encoder_output, hidden, cell = self.encoder(
            source,
            source_lengths
        )

        outputs = torch.zeros(
            batch_size,
            target_length,
            vocab_size,
            device=source.device
        )

        previous_context = torch.zeros(
            batch_size,
            HIDDEN_SIZE * 2,
            device=source.device
        )

        input_token = target[:, 0].unsqueeze(1)

        for t in range(1, target_length):

            output, hidden, cell, previous_context, attention = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_output,
                mask,
                previous_context
            )

            outputs[:, t, :] = output

            best_token = output.argmax(1).unsqueeze(1)

            teacher_force = (
                torch.rand(1).item()
                < teacher_forcing_ratio
            )

            if teacher_force:
                input_token = target[:, t].unsqueeze(1)
            else:
                input_token = best_token

        return outputs


encoder = Encoder(
    VOCAB_SIZE,
    EMBEDDING_SIZE,
    HIDDEN_SIZE,
    DROPOUT
)

decoder = Decoder(
    VOCAB_SIZE,
    EMBEDDING_SIZE,
    HIDDEN_SIZE,
    DROPOUT
)

model = Seq2Seq(
    encoder,
    decoder
).to(device)

total_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", total_params)

# ---------------------------------------------------------------------------------------------------------


#  Training Loop

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)

train_losses = []
valid_losses = []

best_valid_loss = float("inf")

print("\n" + "=" * 60)
print("Starting Training for", NUM_EPOCHS, "Epochs")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    start_time = time.time()

    model.train()
    total_train_loss = 0

    for src, src_lens, tgt in train_loader:

        src = src.to(device)
        src_lens = src_lens.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        output = model(
            src,
            src_lens,
            tgt,
            teacher_forcing_ratio=0.5
        )

        output = output[:, 1:].reshape(
            -1,
            VOCAB_SIZE
        )

        target = tgt[:, 1:].reshape(-1)

        loss = criterion(output, target)

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_valid_loss = 0

    with torch.no_grad():

        for src, src_lens, tgt in valid_loader:

            src = src.to(device)
            src_lens = src_lens.to(device)
            tgt = tgt.to(device)

            output = model(
                src,
                src_lens,
                tgt,
                teacher_forcing_ratio=0.5
            )

            output = output[:, 1:].reshape(
                -1,
                VOCAB_SIZE
            )

            target = tgt[:, 1:].reshape(-1)

            loss = criterion(output, target)

            total_valid_loss += loss.item()

    valid_loss = total_valid_loss / len(valid_loader)

    scheduler.step(valid_loss)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    elapsed = time.time() - start_time

    saved = ""

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "total_params": total_params
            },
            "checkpoints/best_model.pt"
        )

        saved = " <-- BEST"

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Valid Loss: {valid_loss:.4f} | "
        f"PPL: {math.exp(valid_loss):.2f} | "
        f"Time: {elapsed / 60:.2f}m"
        f"{saved}"
    )

with open("logs/training_log.json", "w") as f:

    json.dump(
        {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "best_valid_loss": best_valid_loss,
            "best_perplexity": math.exp(best_valid_loss),
            "total_params": total_params
        },
        f,
        indent=2
    )

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    valid_losses,
    marker="s",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: results/loss_curve.png")

# ---------------------------------------------------------------------------------------------------------

# Load Best Model Checkpoint for Inference

checkpoint = torch.load(
    "checkpoints/best_model.pt",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Loaded best checkpoint from epoch",
    checkpoint["epoch"],
    "with valid loss",
    f"{checkpoint['valid_loss']:.4f}"
)

# ---------------------------------------------------------------------------------------------------------

# Greedy Search

def generate_greedy(source_text, max_length=25):

    model.eval()

    source_ids = (
        [BOS]
        + sp.encode(source_text, out_type=int)
        + [EOS]
    )

    source = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=device
    )

    source_length = torch.tensor(
        [len(source_ids)],
        dtype=torch.long,
        device=device
    )

    mask = source != PAD

    with torch.no_grad():

        encoder_output, hidden, cell = model.encoder(
            source,
            source_length
        )

        previous_context = torch.zeros(
            1,
            HIDDEN_DIM * 2,
            device=device
        )

        current_token = torch.tensor(
            [[BOS]],
            dtype=torch.long,
            device=device
        )

        generated_tokens = []

        for _ in range(max_length):

            output, hidden, cell, previous_context, _ = model.decoder(
                current_token,
                hidden,
                cell,
                encoder_output,
                mask,
                previous_context
            )

            if len(generated_tokens) >= 2:

                last_bigram = (
                    generated_tokens[-2],
                    generated_tokens[-1]
                )

                for i in range(
                    len(generated_tokens) - 2
                ):

                    if (
                        generated_tokens[i],
                        generated_tokens[i + 1]
                    ) == last_bigram:

                        banned_token = generated_tokens[i + 2]
                        output[0, banned_token] = -1e9

            next_token = output.argmax(1).item()

            if next_token == EOS:
                break

            generated_tokens.append(next_token)

            current_token = torch.tensor(
                [[next_token]],
                dtype=torch.long,
                device=device
            )

    return sp.decode(generated_tokens)

# Beam Search

def beam_search(
    source_text,
    beam_width=3,
    max_length=25,
    alpha=0.7
):

    model.eval()

    source_ids = (
        [BOS]
        + sp.encode(source_text, out_type=int)
        + [EOS]
    )

    source = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=device
    )

    source_length = torch.tensor(
        [len(source_ids)],
        dtype=torch.long,
        device=device
    )

    mask = source != PAD

    with torch.no_grad():

        encoder_output, hidden, cell = model.encoder(
            source,
            source_length
        )

        previous_context = torch.zeros(
            1,
            HIDDEN_DIM * 2,
            device=device
        )

        beams = [
            (
                [BOS],
                0.0,
                hidden,
                cell,
                previous_context,
                False
            )
        ]

        completed = []

        for _ in range(max_length):

            candidates = []

            for (
                tokens,
                score,
                h,
                c,
                previous_context,
                done
            ) in beams:

                if done:

                    completed.append(
                        (tokens, score)
                    )

                    continue

                current_token = torch.tensor(
                    [[tokens[-1]]],
                    dtype=torch.long,
                    device=device
                )

                output, next_h, next_c, next_context, _ = model.decoder(
                    current_token,
                    h,
                    c,
                    encoder_output,
                    mask,
                    previous_context
                )

                if len(tokens) >= 3:

                    last_bigram = (
                        tokens[-2],
                        tokens[-1]
                    )

                    for i in range(
                        1,
                        len(tokens) - 2
                    ):

                        if (
                            tokens[i],
                            tokens[i + 1]
                        ) == last_bigram:

                            output[0, tokens[i + 2]] = -1e9

                log_probs = torch.log_softmax(
                    output[0],
                    dim=-1
                )

                top_scores, top_tokens = torch.topk(
                    log_probs,
                    beam_width
                )

                for k in range(beam_width):

                    token = top_tokens[k].item()

                    token_score = top_scores[k].item()

                    new_score = score + token_score

                    new_tokens = tokens + [token]

                    if token == EOS:

                        completed.append(
                            (new_tokens, new_score)
                        )

                    else:

                        candidates.append(
                            (
                                new_tokens,
                                new_score,
                                next_h,
                                next_c,
                                next_context,
                                False
                            )
                        )

            if not candidates:
                break

            candidates.sort(
                key=lambda x:
                x[1] / (len(x[0]) ** alpha),
                reverse=True
            )

            beams = candidates[:beam_width]

            if len(completed) >= beam_width * 2:
                break

        if completed:

            completed.sort(
                key=lambda x:
                x[1] /
                ((len(x[0]) - 1) ** alpha),
                reverse=True
            )

            best_tokens = completed[0][0]

        else:

            best_tokens = beams[0][0]

        output_tokens = [
            token
            for token in best_tokens
            if token not in (BOS, EOS)
        ]

        return sp.decode(output_tokens)

# ---------------------------------------------------------------------------------------------------------

# Heatmap 

def generate_attention_heatmap(
    source_text,
    out_path="results/attention_heatmap.png",
    max_length=25
):

    model.eval()

    source_ids = (
        [BOS]
        + sp.encode(source_text, out_type=int)
        + [EOS]
    )

    source = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=device
    )

    source_length = torch.tensor(
        [len(source_ids)],
        dtype=torch.long,
        device=device
    )

    mask = source != PAD

    with torch.no_grad():

        encoder_output, hidden, cell = model.encoder(
            source,
            source_length
        )

        previous_context = torch.zeros(
            1,
            HIDDEN_DIM * 2,
            device=device
        )

        current_token = torch.tensor(
            [[BOS]],
            dtype=torch.long,
            device=device
        )

        generated_tokens = []
        attentions = []

        for _ in range(max_length):

            output, hidden, cell, previous_context, attention = model.decoder(
                current_token,
                hidden,
                cell,
                encoder_output,
                mask,
                previous_context
            )

            next_token = output.argmax(1).item()

            if next_token == EOS:
                break

            generated_tokens.append(next_token)

            attentions.append(
                attention[0].cpu().numpy()
            )

            current_token = torch.tensor(
                [[next_token]],
                dtype=torch.long,
                device=device
            )

    if not attentions:
        return sp.decode(generated_tokens)

    attention_matrix = torch.tensor(
        attentions
    ).numpy()

    source_pieces = [
        sp.id_to_piece(token)
        for token in source_ids
    ]

    target_pieces = [
        sp.id_to_piece(token)
        for token in generated_tokens
    ]

    plt.figure(
        figsize=(
            max(8, len(source_pieces) * 0.25),
            max(5, len(target_pieces) * 0.35)
        )
    )

    plt.imshow(
        attention_matrix,
        aspect="auto",
        cmap="viridis"
    )

    plt.xticks(
        range(len(source_pieces)),
        source_pieces,
        rotation=90,
        fontsize=8
    )

    plt.yticks(
        range(len(target_pieces)),
        target_pieces,
        fontsize=9
    )

    plt.xlabel("Source Subword Tokens")
    plt.ylabel("Generated Question Tokens")
    plt.title("Attention Alignment Heatmap")

    plt.colorbar(
        label="Attention Weight"
    )

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", out_path)

    return sp.decode(generated_tokens)

# ---------------------------------------------------------------------------------------------------------

# UQA Validation & Wiki-UQA

class UrduWhitespaceTokenizer(tokenizers.Tokenizer):

    def tokenize(self, text):
        return text.split()


urdu_scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=False,
    tokenizer=UrduWhitespaceTokenizer()
)


def evaluate_dataset(
    pairs,
    name="UQA valid",
    num_samples=None
):

    if num_samples is not None:
        eval_pairs = pairs[:num_samples]
    else:
        eval_pairs = pairs

    references = []
    greedy_results = []
    beam_results = []

    start_time = time.time()

    for source, target in eval_pairs:

        references.append(target)

        greedy_results.append(
            generate_greedy(source)
        )

        beam_results.append(
            beam_search(
                source,
                beam_width=3
            )
        )

    greedy_bleu = sacrebleu.corpus_bleu(
        greedy_results,
        [references]
    ).score

    greedy_rouge = sum(
        urdu_scorer.score(
            reference,
            hypothesis
        )["rougeL"].fmeasure
        for reference, hypothesis
        in zip(references, greedy_results)
    ) / len(references)

    greedy_unk = (
        sum(
            result.count("\u2047")
            for result in greedy_results
        )
        /
        max(
            1,
            sum(
                len(result.split())
                for result in greedy_results
            )
        )
    )

    beam_bleu = sacrebleu.corpus_bleu(
        beam_results,
        [references]
    ).score

    beam_rouge = sum(
        urdu_scorer.score(
            reference,
            hypothesis
        )["rougeL"].fmeasure
        for reference, hypothesis
        in zip(references, beam_results)
    ) / len(references)

    beam_unk = (
        sum(
            result.count("\u2047")
            for result in beam_results
        )
        /
        max(
            1,
            sum(
                len(result.split())
                for result in beam_results
            )
        )
    )

    elapsed = time.time() - start_time

    print(
        f"\n--- {name} Results "
        f"({len(eval_pairs)} samples, {elapsed:.1f}s) ---"
    )

    print(
        f"Greedy: BLEU-4 = {greedy_bleu:.2f} | "
        f"ROUGE-L = {greedy_rouge:.4f} | "
        f"<unk>% = {greedy_unk * 100:.2f}%"
    )

    print(
        f"Beam 3: BLEU-4 = {beam_bleu:.2f} | "
        f"ROUGE-L = {beam_rouge:.4f} | "
        f"<unk>% = {beam_unk * 100:.2f}%"
    )

    return {
        "greedy": {
            "bleu": greedy_bleu,
            "rouge": greedy_rouge,
            "unk_pct": greedy_unk * 100
        },
        "beam": {
            "bleu": beam_bleu,
            "rouge": beam_rouge,
            "unk_pct": beam_unk * 100
        },
        "records": list(
            zip(
                [pair[0] for pair in eval_pairs],
                references,
                greedy_results,
                beam_results
            )
        )
    }


print(
    "\nEvaluating first 50 validation pairs "
    "for results/samples.tsv..."
)

val_50_eval = evaluate_dataset(
    valid_pairs,
    name="UQA Validation (50 samples)",
    num_samples=50
)


with open(
    "results/samples.tsv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(
        f,
        delimiter="\t"
    )

    writer.writerow(
        [
            "source",
            "reference",
            "greedy",
            "beam"
        ]
    )

    for row in val_50_eval["records"]:
        writer.writerow(row)


print(
    "Saved: results/samples.tsv with 50 samples"
)


generate_attention_heatmap(
    valid_pairs[0][0],
    out_path="results/attention_heatmap.png"
)


print(
    "\nEvaluating full Wiki-UQA "
    "out-of-domain dataset (177 pairs)..."
)

wiki_eval = evaluate_dataset(
    wiki_pairs,
    name="Wiki-UQA"
)


print(
    "\nEvaluating 500 UQA validation pairs "
    "for Table 3..."
)

val_eval = evaluate_dataset(
    valid_pairs,
    name="UQA Validation (500 samples)",
    num_samples=500
)


model.eval()

val_loss_total = 0
val_batches = 0

with torch.no_grad():

    for src, src_lens, tgt in valid_loader:

        src = src.to(device)
        src_lens = src_lens.to(device)
        tgt = tgt.to(device)

        output = model(
            src,
            src_lens,
            tgt,
            teacher_forcing_ratio=0.5
        )

        loss = criterion(
            output[:, 1:].reshape(-1, VOCAB_SIZE),
            tgt[:, 1:].reshape(-1)
        )

        val_loss_total += loss.item()
        val_batches += 1


mean_v_loss = val_loss_total / val_batches

best_ppl = math.exp(mean_v_loss)

print("Validation Loss:", mean_v_loss)
print("Perplexity:", best_ppl)

# ---------------------------------------------------------------------------------------------------------

# Tables and Figures 
# Model Configuration

with open(
    "results/table2_model_config.tsv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(
        f,
        delimiter="\t"
    )

    writer.writerow(
        ["Property", "Value"]
    )

    writer.writerow(
        [
            "Encoder / decoder type",
            "2-layer BiLSTM / 2-layer LSTM with Bahdanau attention"
        ]
    )

    writer.writerow(
        [
            "Layers / embedding / hidden size",
            f"2 layers / {EMBEDDING_SIZE} / {HIDDEN_SIZE}"
        ]
    )

    writer.writerow(
        [
            "Vocabulary size",
            str(VOCAB_SIZE)
        ]
    )

    writer.writerow(
        [
            "Trainable parameters",
            f"{total_params:,}"
        ]
    )

    writer.writerow(
        [
            "Optimizer, learning rate, schedule",
            "Adam, lr=0.001, ReduceLROnPlateau"
        ]
    )

    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    writer.writerow(
        [
            "Batch size, epochs, GPU",
            f"{BATCH_SIZE}, {NUM_EPOCHS}, {gpu_name}"
        ]
    )

print(
    "Saved: results/table2_model_config.tsv"
)


# Automatic Metrics

with open(
    "results/table3_metrics.tsv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(
        f,
        delimiter="\t"
    )

    writer.writerow(
        [
            "Split",
            "Decoding",
            "BLEU-4",
            "ROUGE-L",
            "PPL",
            "<unk> %"
        ]
    )

    writer.writerow(
        [
            "UQA valid",
            "greedy",
            f"{val_eval['greedy']['bleu']:.2f}",
            f"{val_eval['greedy']['rouge']:.4f}",
            f"{best_ppl:.2f}",
            f"{val_eval['greedy']['unk_pct']:.2f}%"
        ]
    )

    writer.writerow(
        [
            "UQA valid",
            "beam (k=3)",
            f"{val_eval['beam']['bleu']:.2f}",
            f"{val_eval['beam']['rouge']:.4f}",
            f"{best_ppl:.2f}",
            f"{val_eval['beam']['unk_pct']:.2f}%"
        ]
    )

    writer.writerow(
        [
            "Wiki-UQA",
            "greedy",
            f"{wiki_eval['greedy']['bleu']:.2f}",
            f"{wiki_eval['greedy']['rouge']:.4f}",
            "-",
            f"{wiki_eval['greedy']['unk_pct']:.2f}%"
        ]
    )

    writer.writerow(
        [
            "Wiki-UQA",
            "beam (k=3)",
            f"{wiki_eval['beam']['bleu']:.2f}",
            f"{wiki_eval['beam']['rouge']:.4f}",
            "-",
            f"{wiki_eval['beam']['unk_pct']:.2f}%"
        ]
    )

print(
    "Saved: results/table3_metrics.tsv"
)


# Length Histograms

source_lengths = [
    len(source.split())
    for source, target in train_pairs
]

target_lengths = [
    len(target.split())
    for source, target in train_pairs
]


plt.figure(figsize=(7, 4))

plt.hist(
    source_lengths,
    bins=30,
    color="#2b5c8f",
    edgecolor="black"
)

plt.xlabel(
    "Source Sentence Length (words)"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Training Source Sentence Length Distribution"
)

plt.grid(
    axis="y",
    alpha=0.7
)

plt.savefig(
    "results/source_length_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: results/source_length_histogram.png"
)


plt.figure(figsize=(7, 4))

plt.hist(
    target_lengths,
    bins=25,
    color="#107c41",
    edgecolor="black"
)

plt.xlabel(
    "Target Question Length (words)"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Training Target Question Length Distribution"
)

plt.grid(
    axis="y",
    alpha=0.7
)

plt.savefig(
    "results/target_length_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: results/target_length_histogram.png"
)


# Qualitative Samples

qual_records = val_50_eval["records"][:10]

with open(
    "results/qualitative_samples.txt",
    "w",
    encoding="utf-8"
) as f:

    for index, (source, reference, greedy, beam) in enumerate(
        qual_records,
        1
    ):

        f.write(
            f"Sample {index}:\n"
            f"Source: {source}\n"
            f"Reference: {reference}\n"
            f"Greedy: {greedy}\n"
            f"Beam: {beam}\n\n"
        )

print(
    "Saved: results/qualitative_samples.txt"
)

print(
    "\n" + "=" * 60
)

print(
    "COMPLETED SUCCESSFULLY!"
)

print(
    "=" * 60
)
