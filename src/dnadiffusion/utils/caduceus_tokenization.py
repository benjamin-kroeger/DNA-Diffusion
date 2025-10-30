import h5py
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"

# 1. Load remote code (Caduceus defines custom modules like modeling_caduceus.py)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

model.eval()

device = torch.device("cuda")
model.to(device)

def embedd_dna_sequence(seqs:list[str]) -> torch.Tensor:

    inputs = tokenizer(seqs, return_tensors="pt")  # splits chars, uppercases, adds [SEP]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # embeddings from the final hidden layer (per position, incl. [SEP])
    # shape: [batch, seq_len, hidden_dim]
    embeddings = outputs.hidden_states[-1]
    embeddings = embeddings[:,:-1,:]

    return embeddings


def embed_and_save_sequences(df, output_file, chunk_size=200, id_column="dhs_id"):
    """
    Embed DNA sequences and save to HDF5 file with ID-based organization.

    Args:
        df: DataFrame containing 'sequence', 'datalabel', 'TAG', and id_column
        output_file: Path to output HDF5 file
        chunk_size: Number of sequences to process at once
        id_column: Column name to use as keys (default: 'dhs_id')
    """

    with h5py.File(output_file, "w") as h5f:
        # Add file-level attributes
        h5f.attrs["num_samples"] = len(df)
        h5f.attrs["embedding_dim"] = 512
        h5f.attrs["sequence_length"] = 200
        h5f.attrs["id_column"] = id_column

        # Process in chunks
        for start in tqdm(range(0, len(df), chunk_size), desc=f"Embedding sequences"):
            end = min(start + chunk_size, len(df))
            chunk_df = df.iloc[start:end]

            # Embed sequences
            X_chunk = embedd_dna_sequence(
                seqs=chunk_df["sequence"].to_list(),
            ).cpu().numpy()

            # Store each embedding with its ID as the key
            for idx, (_, row) in enumerate(chunk_df.iterrows()):
                dhs_id = str(row[id_column])

                # Create a group for this sample
                grp = h5f.create_group(dhs_id)

                # Store embedding
                grp.create_dataset("embedding", data=X_chunk[idx], dtype="float32")

                # Store metadata as attributes
                grp.attrs["data_label"] = row["data_label"]
                grp.attrs["TAG"] = row["TAG"]


def load_embeddings_from_h5(file_path, n_samples=None):
    """
    Load embeddings and metadata from ID-based HDF5 file.
    """
    with h5py.File(file_path, "r") as h5f:
        ids = list(h5f.keys())

        if n_samples is not None:
            ids = ids[:n_samples]

        num_samples = len(ids)

        # Pre-allocate array
        embeddings = np.zeros((num_samples, 200, 512), dtype="float32")
        datalabels = []
        tags = []

        # Load each embedding
        for idx, dhs_id in enumerate(ids):
            embeddings[idx] = h5f[dhs_id]["embedding"][:]
            datalabels.append(h5f[dhs_id].attrs["data_label"])
            tags.append(h5f[dhs_id].attrs["TAG"])

    return embeddings, np.array(datalabels), np.array(tags), ids
