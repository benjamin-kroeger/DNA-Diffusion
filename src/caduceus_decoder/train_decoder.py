from pathlib import Path
from random import shuffle
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Bio import SeqIO


# =============================================================================
# DECODER ARCHITECTURE
# =============================================================================

class ConvDecoder(nn.Module):
    """
    Fast convolutional decoder for fixed-length sequences.
    Outputs only A, T, G, C (4 classes).
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = 4  # Only A, C, G, T

        # Mapping from Caduceus token IDs to our 4-class system
        # Caduceus: 4=A, 5=C, 6=G, 7=T
        # Ours:     0=A, 1=C, 2=G, 3=T
        self.register_buffer(
            "caduceus_to_idx",
            torch.tensor([
                -1,  # 0: PAD?
                -1,  # 1
                -1,  # 2
                -1,  # 3
                -1,  # 4
                -1,  # 5
                -1,  # 6
                0,  # 7: A
                1,  # 8: C
                2,  # 9: G
                3,  # 10: T
                -1,  # 11: N?
                -1,  # 12
                -1,  # 13
                -1,  # 14
                -1,  # 15
            ], dtype=torch.long)
        )

        # Index to nucleotide character
        self.idx_to_nt = ['A', 'C', 'G', 'T']

        self.proj_in = nn.Linear(input_dim, hidden_dim)

        # Stack of conv layers with residual connections
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size, padding=kernel_size // 2),
                    nn.GLU(dim=1),
                    nn.Dropout(dropout),
                )
            )

        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, seq_len, 4) for A, C, G, T
        """
        x = self.proj_in(x)
        x = x.transpose(1, 2)  # (B, hidden, L)

        for layer in self.layers:
            x = x + layer(x)

        x = x.transpose(1, 2)  # (B, L, hidden)
        x = self.norm(x)
        return self.proj_out(x)

    def convert_targets(self, caduceus_ids: torch.Tensor) -> torch.Tensor:
        """Convert Caduceus token IDs to 0-3 indices."""
        return self.caduceus_to_idx[caduceus_ids.clamp(0, 15)]

    @torch.no_grad()
    def decode(self, embeddings: torch.Tensor) -> list[str]:
        """Convert embeddings to DNA strings (only ACGT)."""
        self.eval()
        logits = self.forward(embeddings)
        preds = logits.argmax(dim=-1)  # (batch, seq_len), values 0-3

        sequences = []
        for i in range(preds.shape[0]):
            seq = ''.join(self.idx_to_nt[idx] for idx in preds[i].cpu().tolist())
            sequences.append(seq)

        return sequences


# =============================================================================
# HDF5 EMBEDDING CACHE
# =============================================================================

class EmbeddingCacheH5:
    """Cache embeddings to HDF5 for efficient storage and loading."""

    def __init__(self, cache_path: str = "./embeddings_cache.h5"):
        self.cache_path = Path(cache_path)

    def cache_exists(self) -> bool:
        return self.cache_path.exists()

    def save(self, embeddings: np.ndarray, targets: np.ndarray):
        """Save embeddings and targets to HDF5."""
        with h5py.File(self.cache_path, 'w') as f:
            # Use compression for smaller file size
            f.create_dataset(
                'embeddings',
                data=embeddings,
                dtype='float32',
                compression='gzip',
                compression_opts=4,
                chunks=(min(100, len(embeddings)), embeddings.shape[1], embeddings.shape[2]),
            )
            f.create_dataset(
                'targets',
                data=targets,
                dtype='int8',  # Small integers, save space
                compression='gzip',
                compression_opts=4,
                chunks=(min(100, len(targets)), targets.shape[1]),
            )

        file_size = self.cache_path.stat().st_size / (1024 ** 2)
        print(f"Cached {len(embeddings)} embeddings to {self.cache_path} ({file_size:.1f} MB)")

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Load embeddings and targets from HDF5."""
        with h5py.File(self.cache_path, 'r') as f:
            embeddings = f['embeddings'][:]
            targets = f['targets'][:]

        print(f"Loaded {len(embeddings)} embeddings from cache")
        return embeddings, targets

    def get_dataset(self) -> 'H5Dataset':
        """Get a dataset that reads directly from H5 (memory efficient)."""
        return H5Dataset(self.cache_path)


class H5Dataset(Dataset):
    """Dataset that reads from HDF5 file - memory efficient for large datasets."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path

        # Get length without loading everything
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['embeddings'])

        self.h5_file = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Lazy open file (needed for multiprocessing)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        emb = torch.from_numpy(self.h5_file['embeddings'][idx].astype(np.float32))
        tgt = torch.from_numpy(self.h5_file['targets'][idx].astype(np.int64))
        return emb, tgt


class CachedEmbeddingDataset(Dataset):
    """In-memory dataset (faster if you have enough RAM)."""

    def __init__(self, embeddings: np.ndarray, targets: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.targets = torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

def generate_embeddings(
    sequences: list[str],
    caduceus_model,
    tokenizer,
    batch_size: int = 20,
    device: str = "cuda",
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate embeddings for all sequences."""

    caduceus_model.eval()
    all_embeddings = []
    all_targets = []

    iterator = range(0, len(sequences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating embeddings")

    with torch.no_grad():
        for i in iterator:
            batch = sequences[i:i + batch_size]

            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = caduceus_model(**inputs, output_hidden_states=True)

            embeddings = outputs.hidden_states[-1][:, :-1, :]  # Remove SEP
            targets = inputs['input_ids'][:, :-1]  # Remove SEP

            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
            all_targets.append(targets.cpu().numpy().astype(np.int8))

    return np.concatenate(all_embeddings), np.concatenate(all_targets)


# =============================================================================
# TRAINING
# =============================================================================

def train_decoder(
    decoder: ConvDecoder,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str = "cuda",
    save_path: str = "decoder.pt",
):
    """Train the decoder."""

    decoder = decoder.to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore invalid tokens

    best_val_acc = 0

    for epoch in range(epochs):
        # Training
        decoder.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for embeddings, targets in pbar:
            embeddings = embeddings.to(device)
            targets = targets.to(device)

            targets = decoder.convert_targets(targets)

            logits = decoder(embeddings)
            loss = criterion(logits.reshape(-1, 4), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1)

            # Only count valid tokens
            valid_mask = targets >= 0
            train_correct += ((preds == targets) & valid_mask).sum().item()
            train_total += valid_mask.sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct / train_total:.2%}" if train_total > 0 else "N/A"
            })

        train_acc = train_correct / train_total if train_total > 0 else 0

        # Validation
        val_acc = 0
        if val_loader:
            decoder.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for embeddings, targets in val_loader:
                    embeddings = embeddings.to(device)
                    targets = targets.to(device)
                    targets = decoder.convert_targets(targets)

                    logits = decoder(embeddings)
                    preds = logits.argmax(dim=-1)

                    valid_mask = targets >= 0
                    val_correct += ((preds == targets) & valid_mask).sum().item()
                    val_total += valid_mask.sum().item()

            val_acc = val_correct / val_total if val_total > 0 else 0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': decoder.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch,
                }, save_path)

        print(f"Epoch {epoch + 1}/{epochs} - Train Acc: {train_acc:.2%} - Val Acc: {val_acc:.2%}")

    print(f"Best validation accuracy: {best_val_acc:.2%}")
    return decoder


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main(
    sequences: list[str],
    caduceus_model,
    tokenizer,
    cache_path: str = "./embeddings_cache.h5",
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: str = "cuda",
    save_path: str = "caduceus_decoder.pt",
    load_to_memory: bool = True,
):
    """
    Full training pipeline with HDF5 caching.

    Args:
        sequences: List of DNA sequences (all same length, e.g., 200bp)
        caduceus_model: Loaded CaduceusForMaskedLM model
        tokenizer: Caduceus tokenizer
        cache_path: HDF5 file path for caching
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        val_split: Fraction for validation
        device: cuda or cpu
        save_path: Where to save trained decoder
        load_to_memory: If True, load all data to RAM (faster). If False, stream from H5.
    """

    cache = EmbeddingCacheH5(cache_path)

    # --- Step 1: Get or generate cached embeddings ---
    if cache.cache_exists():
        print("Loading cached embeddings from H5...")
        embeddings, targets = cache.load()
    else:
        print(f"Generating embeddings for {len(sequences)} sequences...")
        embeddings, targets = generate_embeddings(
            sequences, caduceus_model, tokenizer, batch_size=10, device=device
        )
        cache.save(embeddings, targets)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Targets shape: {targets.shape}")

    # --- Step 2: Create datasets ---
    n_val = int(len(embeddings) * val_split)
    indices = np.random.permutation(len(embeddings))

    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    if load_to_memory:
        train_dataset = CachedEmbeddingDataset(embeddings[train_idx], targets[train_idx])
        val_dataset = CachedEmbeddingDataset(embeddings[val_idx], targets[val_idx])
    else:
        # For very large datasets, use H5 streaming
        # Note: This requires re-saving with split indices
        train_dataset = CachedEmbeddingDataset(embeddings[train_idx], targets[train_idx])
        val_dataset = CachedEmbeddingDataset(embeddings[val_idx], targets[val_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # --- Step 3: Create and train decoder ---
    decoder = ConvDecoder(
        input_dim=512,
        hidden_dim=256,
        num_layers=4,
        kernel_size=7,
        dropout=0.1,
    )

    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    trained_decoder = train_decoder(
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_path=save_path,
    )

    return trained_decoder




def sample_sequences(
    fasta_path: str,
    seqs_per_chr: int =200,
    window_size : int = 800,
    train_split = 0.8
):
    train_test_seqs = []
    for seq_record in SeqIO.parse(fasta_path, "fasta"):

        if not seq_record.id.startswith("chr"):
            continue

        chr_seq = seq_record.seq.upper()
        chr_seq_collection = []
        with tqdm(total=seqs_per_chr,desc=f"Searching seqs on {seq_record.id}") as pbar:
            while len(chr_seq_collection) < seqs_per_chr:

                start_pos = random.randint(0, len(chr_seq) - window_size)
                dna_section = chr_seq[start_pos:start_pos + window_size]
                if "N" in dna_section:
                    continue
                chr_seq_collection.append(str(chr_seq[start_pos:start_pos + window_size]))
                pbar.update(1)

            train_test_seqs.extend(chr_seq_collection)

    shuffle(train_test_seqs)
    train_split_idx = int(len(train_test_seqs) * train_split)

    return train_test_seqs[:train_split_idx], train_test_seqs[train_split_idx:]



# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import random

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Caduceus
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    caduceus = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    caduceus = caduceus.to(device)
    caduceus.eval()

    for p in caduceus.parameters():
        p.requires_grad = False

    # Training data (replace with your real sequences)

    train_sequences, test_seqs = sample_sequences(
        fasta_path="/home/benjaminkroeger/Downloads/GRCh38.p14.genome.fa",
        window_size=3000,
        seqs_per_chr=60
    )


    # Train
    decoder = main(
        sequences=train_sequences,
        caduceus_model=caduceus,
        tokenizer=tokenizer,
        cache_path="./embeddings_cache.h5",
        batch_size=20,
        epochs=10,
        lr=1e-3,
        device=device,
        save_path="caduceus_decoder.pt",
    )



    # --- Test ---
    all_orig = []
    all_recon = []

    for i in range(0, len(test_seqs), 10):
        batch = test_seqs[i:i + 10]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = caduceus(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1][:, :-1, :].to(device)
            recon = decoder.decode(embeddings)

        all_orig.extend(batch)
        all_recon.extend(recon)

    # Evaluate entire test set
    for orig, recon in zip(all_orig, all_recon):
        matches = sum(a == b for a, b in zip(orig, recon))
        print(f"Accuracy: {matches}/{len(orig)} = {matches / len(orig):.1%}")
        print(f"  Orig:  {orig[:50]}...")
        print(f"  Recon: {recon[:50]}...")
        print()



