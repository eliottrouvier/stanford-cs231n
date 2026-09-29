"""
char_rnn_language_model.py - CS231n Lecture 10: Character-Level Language Modeling with RNN
------------------------------------------------------------------------------------------
Implements Andrej Karpathy's iconic "Unreasonable Effectiveness of Recurrent Neural Networks":
1. Recurrent state transitions: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
2. Character tokenization & one-hot / dense embedding.
3. Multi-epoch autoregressive training on a text corpus.
4. Inference with Temperature Sampling: P(c_i) = exp(z_i / T) / sum(exp(z_j / T))
   - Low temperature (0.5): conservative, well-formed words.
   - Balanced temperature (0.8): natural, creative diversity.
   - High temperature (1.2): wild, exploratory.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Chapter 11] Running on device: {device}")

torch.manual_seed(42)
np.random.seed(42)

os.makedirs("11_recurrent_neural_networks/figures", exist_ok=True)


# Classic CS231n excerpt corpus
TRAINING_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
""" * 4 # Repeat for sufficient mini-batches


class CharRNN(nn.Module):
    """
    Vanilla Recurrent Neural Network for character-level prediction.
    """
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.rnn(embeds, hidden)
        logits = self.fc(out)
        return logits, hidden


def sample_text(model, char_to_idx, idx_to_char, seed_text="To be", length=120, temperature=0.8):
    """
    Generates text autoregressively character by character with temperature sampling.
    """
    model.eval()
    chars = [c for c in seed_text if c in char_to_idx]
    if not chars:
        chars = [list(char_to_idx.keys())[0]]
        
    input_indices = [char_to_idx[c] for c in chars]
    curr_input = torch.tensor([input_indices], dtype=torch.long, device=device)
    hidden = None

    generated = "".join(chars)

    with torch.no_grad():
        # Warmup hidden state with prompt
        _, hidden = model(curr_input[:, :-1], hidden)
        last_char = curr_input[:, -1:]

        for _ in range(length):
            logits, hidden = model(last_char, hidden)
            logits = logits.squeeze(0).squeeze(0) / max(temperature, 1e-4)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Sample next char according to probability distribution
            next_idx = np.random.choice(len(probs), p=probs)
            generated += idx_to_char[next_idx]
            last_char = torch.tensor([[next_idx]], dtype=torch.long, device=device)

    return generated


def run_char_rnn_experiment():
    print("\n--- Entraînement du Char-RNN Language Model ---")
    # Build vocabulary
    vocab = sorted(list(set(TRAINING_TEXT)))
    vocab_size = len(vocab)
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}
    print(f"Taille du vocabulaire : {vocab_size} caractères distincts")
    print(f"Longueur du texte d'entraînement : {len(TRAINING_TEXT)} caractères")

    # Prepare sequence dataset (seq_len = 35)
    seq_len = 35
    data_indices = [char_to_idx[ch] for ch in TRAINING_TEXT]
    
    inputs = []
    targets = []
    for i in range(0, len(data_indices) - seq_len, 4):
        inputs.append(data_indices[i:i + seq_len])
        targets.append(data_indices[i + 1:i + seq_len + 1])
        
    X_train = torch.tensor(inputs, dtype=torch.long, device=device)
    Y_train = torch.tensor(targets, dtype=torch.long, device=device)
    print(f"Nombre de séquences d'entraînement : {len(inputs)}")

    # Instantiate model
    model = CharRNN(vocab_size=vocab_size, embedding_dim=64, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    epochs = 40
    batch_size = 32
    loss_history = []

    print("\nEntraînement en cours...")
    for ep in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        batches = 0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            b_x, b_y = X_train[indices], Y_train[indices]

            optimizer.zero_grad()
            logits, _ = model(b_x)
            loss = criterion(logits.view(-1, vocab_size), b_y.view(-1))
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients (CS231n best practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        loss_history.append(avg_loss)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Époque {ep+1:02d}/{epochs:02d} | Cross-Entropy Loss: {avg_loss:.3f}")

    # Generate samples at 3 temperatures
    print("\n--- Échantillonnage de Texte avec Température (Test Time) ---")
    temperatures = [0.4, 0.8, 1.3]
    samples = {}
    for temp in temperatures:
        gen = sample_text(model, char_to_idx, idx_to_char, seed_text="To be", length=140, temperature=temp)
        samples[temp] = gen
        print(f"\n[Température T={temp}] :\n{gen.strip()}\n" + "-"*40)

    # Plot training loss curve and text gallery
    fig = plt.figure(figsize=(14, 6))
    
    # Left: Loss Curve
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(range(1, epochs + 1), loss_history, color="#0284c7", linewidth=2.4, marker="o", markersize=4)
    ax1.set_title("Char-RNN : Décroissance de la Perte (Cross-Entropy)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Époque", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Right: Text Gallery
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    gallery_text = "Échantillons de Génération par Température\n\n"
    for temp, text in samples.items():
        sample_snippet = text.replace('\n', ' ')[:85] + "..."
        desc = "Conservateur / Déterministe" if temp < 0.6 else ("Équilibré / Fluide" if temp < 1.0 else "Exploratoire / Chaotique")
        gallery_text += f"• T={temp} ({desc}) :\n  \"{sample_snippet}\"\n\n"
        
    ax2.text(0.02, 0.85, gallery_text, fontsize=9.5, family="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8fafc", edgecolor="#cbd5e1"))
    ax2.set_title("Génération Conditionnelle selon la Température T", fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_path = "11_recurrent_neural_networks/figures/char_rnn_generation.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\n-> Graphique et galerie sauvegardés : {save_path}")


if __name__ == "__main__":
    t0 = time.time()
    run_char_rnn_experiment()
    print(f"=== Fin du Char-RNN en {time.time() - t0:.2f}s ===")
