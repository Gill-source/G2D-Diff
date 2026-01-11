import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from src.g2d_diff_ce import Condition_Encoder


# ---------------------------
# Load dataset
# ---------------------------
def load_numeric(path):
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


mut = load_numeric("./data/drug_response_data/original_cell2mut.csv")
cna = load_numeric("./data/drug_response_data/original_cell2cna.csv")
cnd = load_numeric("./data/drug_response_data/original_cell2cnd.csv")

meta = pd.read_csv("./data/drug_response_data/DC_drug_response.csv")
meta = meta.dropna(subset=["auc_label", "auc"])

# create cell index mapping
cell_names = pd.read_csv("./data/drug_response_data/original_cell2mut.csv").iloc[:, 1].tolist()
name_to_idx = {n: i for i, n in enumerate(cell_names)}
meta = meta[meta["ccle_name"].isin(name_to_idx.keys())]
meta["cell_idx"] = meta["ccle_name"].apply(lambda x: name_to_idx[x])
# (Optional) subsample for quick evaluation
# Set one of these to limit dataset size; leave both as None to use full set.
SAMPLE_FRAC = None   # e.g., 0.1 for 10%
SAMPLE_SIZE = None  # e.g., 500 rows
RANDOM_STATE = 42

if SAMPLE_FRAC is not None:
    meta = meta.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)
elif SAMPLE_SIZE is not None:
    meta = meta.sample(n=min(SAMPLE_SIZE, len(meta)), random_state=RANDOM_STATE).reset_index(drop=True)


# ---------------------------
# Load pre-trained encoder + regressor
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resolve checkpoint path (relative to this file's directory)
base_dir = os.path.dirname(os.path.abspath(__file__))
preferred_ckpts = [
    os.path.join(base_dir, "condition_encoder_pretrained.ckpt_epoch_13.ckpt"),
    os.path.join(base_dir, "condition_encoder_pretrained.ckpt_epoch_2.ckpt"),
    os.path.join(base_dir, "condition_encoder_pretrained.ckpt_epoch_1.ckpt"),
    os.path.join(base_dir, "condition_encoder_pretrained.ckpt"),
]

ckpt_path = None
for p in preferred_ckpts:
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path is None:
    candidates = glob.glob(os.path.join(base_dir, "condition_encoder_pretrained.ckpt_epoch_*.ckpt"))
    if candidates:
        def get_epoch(name):
            base = os.path.basename(name)
            try:
                return int(base.split("_epoch_")[-1].replace(".ckpt", ""))
            except Exception:
                return -1
        candidates = sorted(candidates, key=get_epoch)
        ckpt_path = candidates[-1]
        print(f"⚠️ Using latest found checkpoint: {ckpt_path}")
    else:
        raise FileNotFoundError(
            "No condition encoder checkpoint found "
            "(looked for condition_encoder_pretrained.ckpt[,_epoch_*.ckpt])."
        )

ckpt = torch.load(ckpt_path, map_location=device)

model = Condition_Encoder(
    num_of_genotypes=3, num_of_dcls=5, num_of_genes=720, gene_emb_size=128,
    device=device, neighbor_info=True
).to(device)
# pad adjacency if NeST adj is 718x718
if model.gene_adj.shape[0] != 720:
    new_adj = torch.zeros((720, 720), device=device, dtype=model.gene_adj.dtype)
    h, w = model.gene_adj.shape
    new_adj[:h, :w] = model.gene_adj
    model.gene_adj = new_adj

model.load_state_dict(ckpt["condition_state_dict"])
model.eval()

regressor = torch.nn.Sequential(
    torch.nn.Linear(128, 128),
    torch.nn.GELU(),
    torch.nn.Linear(128, 1),
).to(device)
# load trained regressor weights if present
if "regressor_state_dict" in ckpt:
    regressor.load_state_dict(ckpt["regressor_state_dict"])
    print(f"✅ Loaded regressor weights from {ckpt_path}")
else:
    print("⚠️ regressor_state_dict not found in checkpoint — using randomly initialized regressor.")


# ---------------------------
# Evaluate
# ---------------------------
preds = []
trues = []

with torch.no_grad():
    for _, row in tqdm(meta.iterrows(), total=len(meta)):

        cell_idx = row["cell_idx"]
        cls = torch.LongTensor([row["auc_label"]]).to(device)
        auc_true = float(row["auc"])

        # prepare genotype
        def pad(x):
            x = np.pad(x, (0, max(0, 720-len(x))))[:720]
            # add batch dim to match Condition_Encoder input shape (B, genes)
            return torch.FloatTensor(x).unsqueeze(0).to(device)

        genotype = {
            "MUT": pad(mut.iloc[cell_idx].values),
            "CNA": pad(cna.iloc[cell_idx].values),
            "CND": pad(cnd.iloc[cell_idx].values),
        }

        cond_input = {"genotype": genotype, "class": cls}

        _, latent, _, _ = model(cond_input)
        auc_pred = regressor(latent).item()

        preds.append(auc_pred)
        trues.append(auc_true)


# ---------------------------
# Compute metrics
# ---------------------------
rmse = np.sqrt(mean_squared_error(trues, preds))
mae = mean_absolute_error(trues, preds)
r2 = r2_score(trues, preds)
pcc, _ = pearsonr(trues, preds)

print("\n====== Prediction Performance (Your Reproduction) ======")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"R²   = {r2:.4f}")
print(f"PCC  = {pcc:.4f}")

# Save
df = pd.DataFrame({"true": trues, "pred": preds})
df.to_csv("pred_vs_true_auc.csv", index=False)
