import h5py
import pandas as pd

from train_decoder import ConvDecoder
import torch


def load_decoder(checkpoint_path):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Instantiate the UNet model
    model = ConvDecoder(
        input_dim=512,
        hidden_dim=256,
        num_layers=4,
        kernel_size=7,
        dropout=0.1,
    )

    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to("cuda")
    return model


num_to_key = {1: 'GM12878_ENCLB441ZZZ', 2: 'HepG2_ENCLB029COU', 3: 'K562_ENCLB843GMH', 4: 'hESCT0_ENCLB449ZZZ'}


def decode_embeddings(h5_path, decoder):
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        synth_dna_seqs = []
        for key in keys:
            synth_embedding = torch.tensor(f[key]["embedding"][:]).squeeze().transpose(1, 2)
            if len(synth_embedding.shape) == 2:
                synth_embedding = synth_embedding.transpose(0,1)
            elif len(synth_embedding.shape) == 3:
                synth_embedding = synth_embedding.transpose(1,2)
            synth_embedding = synth_embedding.to("cuda")

            dna_seq = decoder.decode(synth_embedding)
            synth_dna_seqs.extend(dna_seq)

    return synth_dna_seqs

def decode_my_embedding_format(h5_path, decoder):
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        synth_dna_seqs = []
        for key in keys:
            synth_embedding = torch.tensor(f[key]["embedding"][:]).unsqueeze(dim=0)
            synth_embedding = synth_embedding.to("cuda")

            dna_seq = decoder.decode(synth_embedding)
            synth_dna_seqs.extend(dna_seq)

    return synth_dna_seqs


def generate_synth_dna_df(filepaths: list[str], model):
    synth_dna_seqs = []
    tags = []
    ids = []
    data_label = []

    _id_count = 0
    for filepath in filepaths:
        new_seqs = decode_embeddings(filepath, model)
        synth_dna_seqs.extend(new_seqs)
        ids.extend([f"foundation_synth_{i}" for i in list(range(_id_count, _id_count + len(new_seqs)))])
        _id_count = _id_count + len(new_seqs)
        data_label.extend(["foundation_model_synth"] * len(new_seqs))
        cell_type_num = filepath.rstrip(".h5").split("_")[-1]
        tags.extend([num_to_key[int(cell_type_num)]] * len(new_seqs))

    pd.DataFrame({"sequence": synth_dna_seqs, "TAG": tags, "dhs_id": ids, "data_label": data_label}).to_csv("synth_dna_seqs.csv", index=False)


if __name__ == "__main__":
    model = load_decoder("/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/src/caduceus_decoder/caduceus_decoder_done.pt")
    generate_synth_dna_df(
        ["/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/data/final_analysis_data/embeddings/caduceus/dhs_test_seqs.h5",
         ], model)
