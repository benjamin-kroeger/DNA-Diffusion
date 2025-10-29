import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

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


