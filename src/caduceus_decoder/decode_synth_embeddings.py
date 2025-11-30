import h5py

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

def decode_embeddings(h5_path, decoder):

    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())

        for key in keys:
            synth_embedding = torch.tensor(f[key]["embedding"][-1,:,:,:]).squeeze().transpose(0,1)
            synth_embedding = synth_embedding.to("cuda")
            synth_embedding = synth_embedding.unsqueeze(dim=0)

            dna_seq = decoder.decode(synth_embedding)
            print(dna_seq)


if __name__ == "__main__":
    model = load_decoder("/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/src/caduceus_decoder/caduceus_decoder_done.pt")
    decode_embeddings("/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/sampled_embedd_1.h5", model)
    decode_embeddings("/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/sampled_embedd_2.h5", model)
    decode_embeddings("/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/sampled_embedd_3.h5", model)
    decode_embeddings("/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/sampled_embedd_4.h5", model)

