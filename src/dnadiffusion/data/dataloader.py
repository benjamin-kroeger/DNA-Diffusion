import os
import pickle
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dnadiffusion.utils.caduceus_tokenization import embed_and_save_sequences, load_embeddings_from_h5


def get_dataset(
    data_path: str,
    saved_partition_path: str,
    load_prepartitioning: bool,
    debug: bool,
    load_embeddings: bool = True,
    embedding_save_path: str | None = None,
    foundation_model: str | None = None,

) -> tuple[tuple[Dataset, Dataset, list[int], dict[int, str]], torch.Tensor, torch.Tensor]:
    encode_data = load_data(
        data_path,
        saved_partition_path,
        load_prepartitioning,
        load_embeddings=load_embeddings,
        embedding_save_path=embedding_save_path,
        foundation_model=foundation_model,
    )
    if debug:
        x_data = encode_data["X_train"][:1]
        y_data = encode_data["x_train_cell_type"][:1]
        x_val_data = encode_data["X_val"][:1]
        y_val_data = encode_data["x_val_cell_type"][:1]

    else:
        x_data = encode_data["X_train"]
        y_data = encode_data["x_train_cell_type"]
        x_val_data = encode_data["X_val"]
        y_val_data = encode_data["x_val_cell_type"]

    cell_num_list = encode_data["cell_types"]
    numeric_to_tag_dict = encode_data["numeric_to_tag"]

    train_data = SequenceDataset(x_data, y_data)
    val_data = SequenceDataset(x_val_data, y_val_data)

    return (train_data, val_data, cell_num_list, numeric_to_tag_dict), encode_data["train_mu"], encode_data["train_sd"]


def get_dataset_for_sampling(
    data_path: str,
    saved_data_path: str,
    load_saved_data: bool,
    debug: bool,
    cell_types: str | list[str] | None = None,
) -> tuple[Dataset, Dataset, list[int], dict[int, str]]:
    train_data, val_data, cell_num_list, numeric_to_tag_dict = get_dataset(
        data_path,
        saved_data_path,
        load_saved_data,
        debug,
    )

    if cell_types is None:
        return train_data, val_data, cell_num_list, numeric_to_tag_dict

    if isinstance(cell_types, str):
        if "," in cell_types:
            cell_types = [ct.strip() for ct in cell_types.split(",")]
        else:
            cell_types = [cell_types]

    tag_to_numeric = {tag: num for num, tag in numeric_to_tag_dict.items()}

    filtered_cell_nums = []
    for cell_type_query in cell_types:
        if cell_type_query in tag_to_numeric:
            filtered_cell_nums.append(tag_to_numeric[cell_type_query])
        else:
            matches = [tag for tag in tag_to_numeric.keys() if cell_type_query.lower() in tag.lower()]
            if len(matches) == 1:
                filtered_cell_nums.append(tag_to_numeric[matches[0]])
                print(f"Matched '{cell_type_query}' to '{matches[0]}'")
            elif len(matches) > 1:
                print(f"Warning: '{cell_type_query}' matches multiple cell types: {matches}. Please be more specific.")
            else:
                print(
                    f"Warning: Cell type '{cell_type_query}' not found in dataset. Available types: {list(tag_to_numeric.keys())}"
                )

    if not filtered_cell_nums:
        raise ValueError(f"No valid cell types found. Available types: {list(tag_to_numeric.keys())}")

    return train_data, val_data, filtered_cell_nums, numeric_to_tag_dict


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    pin_memory: bool,
) -> tuple[DataLoader, Any]:
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    dataloader = DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader, sampler


def _compute_mean_std(X_train_embed, output_dir):
    # X_train_embed: numpy array (N,D, L), float32
    embedd_dim = X_train_embed.shape[1]
    X = torch.from_numpy(X_train_embed)  # (N, 48, 200)
    mu = X.mean(dim=(0, 2))
    sd = X.std(dim=(0, 2), unbiased=False)
    sd = sd.clamp_min(1e-6)

    # reshape to broadcast with (B, 1, D, L)
    mu = mu.view(1, 1, embedd_dim, 1)
    sd = sd.view(1, 1, embedd_dim, 1)

    torch.save({"mu": mu, "sd": sd}, os.path.join(output_dir, "embed_whiten_stats.pt"))
    return mu, sd


def load_data(
    data_path: str,
    saved_partition_path: str,
    load_prepartitioning: bool,
    load_embeddings: bool,
    embedding_save_path: str,
    foundation_model: str,
):
    # Preprocessing data
    if load_prepartitioning:
        with open(saved_partition_path, "rb") as f:
            encode_data = pickle.load(f)
    else:
        encode_data = preprocess_data(data_path, saved_partition_path)

    # Creating sequence dataset
    df = encode_data["train_df"]
    val_df = encode_data["validation_df"]

    req_embedding_save_path = os.path.join(embedding_save_path, foundation_model)
    train_embed_file = os.path.join(req_embedding_save_path, "train_embeddings.h5")
    val_embed_file = os.path.join(req_embedding_save_path, "val_embeddings.h5")
    if not load_embeddings:
        embed_and_save_sequences(
            df=df,
            chunk_size=200,
            output_file=train_embed_file,
        )
        embed_and_save_sequences(
            df=val_df,
            chunk_size=200,
            output_file=val_embed_file
        )

    X_train,_,_ = load_embeddings_from_h5(train_embed_file)
    X_train = X_train.transpose(0, 2, 1)

    X_val,_,_ = load_embeddings_from_h5(val_embed_file)
    X_val = X_val.transpose(0, 2, 1)

    if not load_embeddings:
        # compute and store std and mean
        train_mu, train_sd = _compute_mean_std(X_train, req_embedding_save_path)
    else:
        mu_sd_dict = torch.load(os.path.join(req_embedding_save_path, "embed_whiten_stats.pt"))
        train_mu = mu_sd_dict["mu"]
        train_sd = mu_sd_dict["sd"]

    # Creating labels
    tag_to_numeric = {x: n for n, x in enumerate(df["TAG"].unique(), 1)}
    numeric_to_tag = dict(enumerate(df["TAG"].unique(), 1))
    cell_types = list(numeric_to_tag.keys())
    x_train_cell_type = torch.tensor([tag_to_numeric[x] for x in df["TAG"]])

    # Creating labels for test
    x_val_cell_type = torch.tensor([tag_to_numeric[x] for x in val_df["TAG"]])

    # Collecting variables into a dict
    encode_data_dict = {
        "tag_to_numeric": tag_to_numeric,
        "numeric_to_tag": numeric_to_tag,
        "cell_types": cell_types,
        "X_train": X_train,
        "X_val": X_val,
        "x_train_cell_type": x_train_cell_type,
        "x_val_cell_type": x_val_cell_type,
        "train_mu": train_mu,
        "train_sd": train_sd
    }

    return encode_data_dict


def preprocess_data(input_data_path: str, output_path: str | None = None) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(input_data_path, sep="\t")

    df_train = df[(df["chr"] != "chr1") & (df["chr"] != "chr2")].reset_index(drop=True)
    df_validation = df[df["chr"] == "chr2"].reset_index(drop=True)
    df_test = df[df["chr"] == "chr1"].reset_index(drop=True)

    encode_data = {
        "train_df": df_train,
        "validation_df": df_validation,
        "test_df": df_test,
    }

    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(encode_data, f)

    return encode_data


class SequenceDataset(Dataset):
    def __init__(
        self,
        seqs: np.ndarray,
        c: torch.Tensor,
        transform: T.Compose = T.Compose([T.ToTensor()]),
    ):
        "Initialization"
        self.seqs = seqs
        self.c = c
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.seqs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        image = self.seqs[index]

        if self.transform:
            x = self.transform(image)
        else:
            x = image

        y = self.c[index]

        return x, y


if __name__ == "__main__":
    pass
