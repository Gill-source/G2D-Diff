import os
import glob
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau

# ---------------------------
# internal imports
# ---------------------------
from src.g2d_diff_ce import Condition_Encoder
from src.g2d_diff_diff import Diffusion

# SA scorer
try:
    from src.sascorer import calculateScore as calc_sa
except:
    from vae_package.sascorer import calculateScore as calc_sa

# VAE LSTM decoder modules
from vae_package import vocab
from vae_package.vae_lstm_model import RNNVAE
from vae_package.vae_lstm_tool import RNNVAESampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================
# 0) Load numeric CSV
# ===========================================================
def load_numeric(path):
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

mut = load_numeric("./data/drug_response_data/original_cell2mut.csv")
cna = load_numeric("./data/drug_response_data/original_cell2cna.csv")
cnd = load_numeric("./data/drug_response_data/original_cell2cnd.csv")

meta = pd.read_csv("./data/drug_response_data/DC_drug_response.csv")
meta = meta.dropna(subset=["auc_label", "auc"])

# cell mapping
cell_names = pd.read_csv("./data/drug_response_data/original_cell2mut.csv").iloc[:, 1].tolist()
name_to_idx = {n: i for i, n in enumerate(cell_names)}
meta = meta[meta["ccle_name"].isin(name_to_idx.keys())]
meta["cell_idx"] = meta["ccle_name"].apply(lambda x: name_to_idx[x])

# ===========================================================
# 1) Load Condition Encoder
# ===========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_candidates = glob.glob(os.path.join(base_dir, "condition_encoder_pretrained_epoch_*.ckpt"))
assert len(ckpt_candidates) > 0, "❌ No condition encoder ckpt found"

def extract_ep(path):
    try:
        return int(os.path.basename(path).split("_epoch_")[-1].replace(".ckpt", ""))
    except:
        return -1

ckpt_candidates = sorted(ckpt_candidates, key=extract_ep)
cond_ckpt_path = ckpt_candidates[-1]
print(f"✅ Loaded Condition Encoder: {cond_ckpt_path}")

cond_ckpt = torch.load(cond_ckpt_path, map_location=device)

encoder = Condition_Encoder(
    num_of_genotypes=3,
    num_of_dcls=5,
    num_of_genes=720,
    gene_emb_size=128,
    device=device,
    neighbor_info=True
).to(device).eval()

# pad adj
if encoder.gene_adj.shape[0] != 720:
    adj = torch.zeros((720,720), device=device)
    h,w = encoder.gene_adj.shape
    adj[:h,:w] = encoder.gene_adj
    encoder.gene_adj = adj

encoder.load_state_dict(cond_ckpt["condition_state_dict"])
encoder.eval()

# pretrained regressor
regressor = torch.nn.Sequential(
    torch.nn.Linear(128,128),
    torch.nn.GELU(),
    torch.nn.Linear(128,1)
).to(device)

if "regressor_state_dict" in cond_ckpt:
    regressor.load_state_dict(cond_ckpt["regressor_state_dict"])
    print("✅ Loaded pretrained regressor")
else:
    print("⚠ regressor_state_dict NOT found — WARNING: prediction accuracy will be random")

regressor.eval()

# ===========================================================
# 2) Load trained Diffusion model
# ===========================================================
diff_ckpt_path = "diffusion_models/auto_save_epoch_360.ckpt"
print(f"📦 Loading Diffusion ckpt: {diff_ckpt_path}")

d_ckpt = torch.load(diff_ckpt_path, map_location=device)

diff_model = Diffusion(training=False).to(device).eval()
diff_model.load_state_dict(d_ckpt["diffusion_state_dict"])

# ===========================================================
# 3) Load VAE LSTM decoder
# ===========================================================
ENABLE_GENERATION = True

# Try multiple vocab paths
possible_vocab_files = [
    "old_vocab.txt",
    "old_tokens.txt",
    "old_vocab_revised.txt",
]

vocab_path = None
for f in possible_vocab_files:
    path = os.path.join(base_dir, "vae_package", f)
    if os.path.exists(path):
        vocab_path = path
        break

vae_ckpt_path = os.path.join(base_dir, "vae_package", "250_lstm09.ckpt")

if vocab_path is None or not os.path.exists(vae_ckpt_path):
    print("⚠ VAE vocab/ckpt missing → generation disabled")
    ENABLE_GENERATION = False

if ENABLE_GENERATION:
    print(f"📦 Loading VAE-LSTM decoder with vocab: {vocab_path}")

    vo = vocab.Vocabulary(init_from_file=vocab_path)
    smtk = vocab.SmilesTokenizer(vo)

    vae_params = {"CHAR_DICT": vo.vocab}
    vae = RNNVAE(vo, smtk, params=vae_params, name="vae_decoder", device=device)
    vae.load(vae_ckpt_path, device)

    vae.model.eval()
    sampler = RNNVAESampler(vae, vo, batch_size=32)

# ===========================================================
# Helper
# ===========================================================
def pad_gene(x):
    return torch.FloatTensor(np.pad(x, (0, max(0,720-len(x))))[:720]).unsqueeze(0).to(device)

# ===========================================================
# 4) AUC Prediction Performance
# ===========================================================
preds = []
trues = []

print("🔍 Evaluating AUC prediction...")

with torch.no_grad():
    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        
        cell_idx = row["cell_idx"]
        cls = torch.LongTensor([row["auc_label"]]).to(device)

        genotype = {
            "MUT": pad_gene(mut.iloc[cell_idx].values),
            "CNA": pad_gene(cna.iloc[cell_idx].values),
            "CND": pad_gene(cnd.iloc[cell_idx].values),
        }

        latent = encoder({"genotype": genotype, "class": cls})[1]
        auc_pred = regressor(latent).item()

        preds.append(auc_pred)
        trues.append(float(row["auc"]))

# metrics
rmse = np.sqrt(mean_squared_error(trues, preds))
mae = mean_absolute_error(trues, preds)
r2 = r2_score(trues, preds)
pcc, _ = pearsonr(trues, preds)
scc, _ = spearmanr(trues, preds)
kt, _ = kendalltau(trues, preds)

print("\n======= PREDICTION PERFORMANCE =======")
print(f"RMSE       = {rmse:.4f}")
print(f"MAE        = {mae:.4f}")
print(f"R²         = {r2:.4f}")
print(f"Pearson r  = {pcc:.4f}")
print(f"Spearman   = {scc:.4f}")
print(f"KendallTau = {kt:.4f}")

