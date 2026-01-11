import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob
import collections

from src.g2d_diff_ce import Condition_Encoder

# --------------------------------------------------------------------
# PyTorch 1.11 + Python 3.10: ensure collections.Container is present
# (required by torch.cuda.nccl reduce_add_coalesced)
# --------------------------------------------------------------------
import collections.abc
for _name in ["Container", "Iterable", "Mapping", "MutableMapping", "Sequence", "MutableSequence", "Sized"]:
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))


# ==========================================
# Dataset
# ==========================================
class GenotypeAucDataset(Dataset):
    def __init__(self, cell2mut, cell2cna, cell2cnd, meta, target_genes=720):

        self.meta = meta.reset_index(drop=True)
        self.mut = torch.tensor(cell2mut.values, dtype=torch.float32)
        self.cna = torch.tensor(cell2cna.values, dtype=torch.float32)
        self.cnd = torch.tensor(cell2cnd.values, dtype=torch.float32)
        self.target_genes = target_genes

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):

        row = self.meta.iloc[idx]
        cell_idx = row["cell_idx"]
        drug_class = int(row["class"])
        auc = torch.tensor(row["auc"], dtype=torch.float32)

        def pad(x):
            if len(x) < self.target_genes:
                x = torch.cat([x, torch.zeros(self.target_genes - len(x))])
            return x[:self.target_genes]

        genotype = {
            "MUT": pad(self.mut[cell_idx]),
            "CNA": pad(self.cna[cell_idx]),
            "CND": pad(self.cnd[cell_idx]),
        }

        cls = torch.LongTensor([drug_class])
        return genotype, cls, auc


# ==========================================
# Pretraining (Multi-GPU 적용)
# ==========================================
def pretrain_condition_encoder(
        mut_path,
        cna_path,
        cnd_path,
        drug_response_path,
        save_path="condition_encoder_pretrained.ckpt",
        batch_size=64,
        lr=1e-4,
        epochs=50,
        device="cuda"
):

    # --------------------------
    # Load data
    # --------------------------
    print("Loading genotype data...")

    def load_numeric_with_names(path):
        df_raw = pd.read_csv(path)
        cell_names = df_raw.iloc[:, 1].tolist()
        df = df_raw.drop(columns=df_raw.columns[:2])
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return df, cell_names

    cell2mut, names_mut = load_numeric_with_names(mut_path)
    cell2cna, names_cna = load_numeric_with_names(cna_path)
    cell2cnd, names_cnd = load_numeric_with_names(cnd_path)

    if not (names_mut == names_cna == names_cnd):
        raise ValueError("Cell name lists mismatch!")

    meta = pd.read_csv(drug_response_path)
    meta = meta.dropna(subset=["auc"])

    name_to_idx = {n: i for i, n in enumerate(names_mut)}
    meta = meta[meta["ccle_name"].isin(name_to_idx.keys())].reset_index(drop=True)
    meta["cell_idx"] = meta["ccle_name"].apply(lambda x: name_to_idx[x])
    meta = meta.rename(columns={"auc_label": "class"})

    dataset = GenotypeAucDataset(cell2mut, cell2cna, cell2cnd, meta)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # --------------------------
    # 모델 정의 (DP 적용)
    # --------------------------
    cond_encoder = Condition_Encoder(
        num_of_genotypes=3,
        num_of_dcls=5,
        num_of_genes=720,
        device=device,
        neighbor_info=True
    )

    if cond_encoder.gene_adj.shape[0] != 720:
        new_adj = torch.zeros((720, 720), dtype=cond_encoder.gene_adj.dtype)
        h, w = cond_encoder.gene_adj.shape
        new_adj[:h, :w] = cond_encoder.gene_adj
        cond_encoder.gene_adj = new_adj

    regressor = nn.Sequential(
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, 1)
    )

    # --- DataParallel 적용 (GPU 여러 장 사용) ---
    cond_encoder = nn.DataParallel(cond_encoder).to(device)
    regressor = nn.DataParallel(regressor).to(device)

    optimizer = optim.AdamW(
        list(cond_encoder.parameters()) + list(regressor.parameters()),
        lr=lr
    )
    criterion = nn.MSELoss()

    # --------------------------
    # Resume latest checkpoint
    # --------------------------
    base_path = save_path if save_path.endswith(".ckpt") else f"{save_path}.ckpt"
    base_prefix = base_path.replace(".ckpt", "")
    ckpt_candidates = glob.glob(f"{base_prefix}_epoch_*.ckpt")

    start_epoch = 0
    if ckpt_candidates:
        ckpt_candidates = sorted(ckpt_candidates, key=lambda x: int(x.split("_epoch_")[1].replace(".ckpt", "")))
        latest_ckpt = ckpt_candidates[-1]
        ckpt = torch.load(latest_ckpt, map_location=device)

        cond_encoder.module.load_state_dict(ckpt["condition_state_dict"])
        regressor.module.load_state_dict(ckpt["regressor_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = int(latest_ckpt.split("_epoch_")[1].replace(".ckpt", ""))
        print(f"🔄 Resuming from {latest_ckpt}")

    print("🚀 Start TRUE Condition Encoder Pretraining (AUC regression) ...")

    # ==========================================
    # Training Loop
    # ==========================================
    for epoch in range(start_epoch, epochs):

        cond_encoder.train()
        regressor.train()
        epoch_loss = 0

        for genotype, cls, auc in tqdm(loader):

            for k in genotype.keys():
                genotype[k] = genotype[k].to(device)

            cls = cls.to(device)
            auc = auc.to(device).view(-1, 1)

            cond_input = {"genotype": genotype, "class": cls}

            _, latent, _, _ = cond_encoder(cond_input)
            pred_auc = regressor(latent)

            loss = criterion(pred_auc, auc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss = {epoch_loss/len(loader):.6f}")

        # Save checkpoint
        save_path_epoch = base_prefix + f"_epoch_{epoch+1}.ckpt"
        torch.save({
            "condition_state_dict": cond_encoder.module.state_dict(),
            "regressor_state_dict": regressor.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, save_path_epoch)

    print("\n🎉 Pretraining Completed!")
    print(f"Saved pretrained encoder to: {save_path}")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    pretrain_condition_encoder(
        mut_path="./data/drug_response_data/original_cell2mut.csv",
        cna_path="./data/drug_response_data/original_cell2cna.csv",
        cnd_path="./data/drug_response_data/original_cell2cnd.csv",
        drug_response_path="./data/drug_response_data/DC_drug_response.csv",
        epochs=20,
        batch_size=48,
        save_path="condition_encoder_pretrained.ckpt"
    )
