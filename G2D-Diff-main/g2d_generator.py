import torch
import pandas as pd
import numpy as np

from g2d_generator import G2DGenerator
from figure4_generate import generate_figure4

from g2d_diff_ce import Condition_Encoder
from g2d_diff_diff import Diffusion

from vae_lstm_model import RNNVAE
from vae_lstm_tool import RNNVAESampler
from vae_package import vocab


# -----------------------------
# Environment
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# 1. Load Diffusion Model
# -----------------------------
ckpt_path = "diffusion_models/auto_save_epoch_360.ckpt"
print("Loading diffusion checkpoint:", ckpt_path)

ckpt = torch.load(ckpt_path, map_location=device)

diffusion = Diffusion().to(device)
diffusion.load_state_dict(ckpt["diffusion_state_dict"])
diffusion.eval()


# -----------------------------
# 2. Load Condition Encoder
# -----------------------------
cond_encoder = Condition_Encoder(
    num_of_genotypes=3,
    num_of_dcls=5,
    num_of_genes=720,
    gene_emb_size=128,
    device=device,
    neighbor_info=True
).to(device)
cond_encoder.eval()

# Fix adjacency padding if needed
if cond_encoder.gene_adj.shape[0] != 720:
    new_adj = torch.zeros((720,720), device=device)
    h, w = cond_encoder.gene_adj.shape
    new_adj[:h,:w] = cond_encoder.gene_adj
    cond_encoder.gene_adj = new_adj


# -----------------------------
# 3. Load VAE + Sampler
# -----------------------------
print("Loading VAE model...")

# load vocab/tokens
vo = vocab.Vocabulary("bongsung_tokens.txt")
smtk = vocab.SmilesTokenizer(vo)

vae_ckpt_path = "vae.ckpt"   # 필요시 수정
rnn_vae = RNNVAE(
    vo=vo,
    smtk=smtk,
    device=device,
    load_fn=vae_ckpt_path
)

vae_sampler = RNNVAESampler(rnn_vae, vo, batch_size=32)


# -----------------------------
# 4. Prepare Genotype (Pick one cell)
# -----------------------------
meta = pd.read_csv("./data/drug_response_data/DC_drug_response.csv")
cell2mut = pd.read_csv("./data/drug_response_data/original_cell2mut.csv")

# pick first cell
cell_index = 0
mut = pd.to_numeric(cell2mut.iloc[cell_index], errors="coerce").fillna(0).values
mut = np.pad(mut, (0, max(0, 720-len(mut))))[:720]

mut_tensor = torch.FloatTensor(mut).unsqueeze(0).to(device)

genotype = {
    "MUT": mut_tensor,
    "CNA": mut_tensor,
    "CND": mut_tensor
}


# -----------------------------
# 5. Create the Full Generator
# -----------------------------
generator = G2DGenerator(
    diffusion_model=diffusion,
    cond_encoder=cond_encoder,
    rnn_vae=rnn_vae,
    sampler=vae_sampler,
    device=device
)

print("G2D-Diff Generator initialized.")


# -----------------------------
# 6. Generate Figure 4 (class 0–4)
# -----------------------------
for cls in range(5):
    print(f"\nGenerating Figure 4 for Class {cls}...")
    generate_figure4(generator, genotype, cls, num=4, device=device)

print("\nAll Figure 4 images generated and saved successfully.")
