import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.utils import shuffle
import copy
import accelerate
from accelerate import Accelerator
import pickle
import glob

from src.utils.g2d_diff_genodrug_dataset import *
from src.g2d_diff_ce import *
from src.g2d_diff_diff import *

from einops import rearrange, repeat, reduce
from functools import partial
import math
import torch.distributed as dist


###########################################################################
# 🔥 Resume 기능을 위해 필요한 헬퍼 함수
###########################################################################
def get_latest_checkpoint(folder="diffusion_models"):
    """폴더에서 가장 최근 epoch ckpt 파일 자동 탐색"""
    if not os.path.exists(folder):
        return None

    ckpts = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    if len(ckpts) == 0:
        return None

    # 파일명 형식: auto_save_epoch_100.ckpt → 숫자만 파싱
    def extract_epoch(name):
        parts = name.replace(".ckpt", "").split("_")
        return int(parts[-1])

    ckpts_sorted = sorted(ckpts, key=lambda x: extract_epoch(x))
    latest = ckpts_sorted[-1]
    return os.path.join(folder, latest)

###########################################################################


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ##############
    # Data load
    ##############
    PREDIFINED_GENOTYPES = ['mut', 'cna', 'cnd']

    nci_data = pd.read_csv("./data/drug_response_data/DC_drug_response.csv").dropna()

    val_cell = ['EKVX_LUNG', 'SKMEL28_SKIN', 'SKOV3_OVARY', 'NCIH226_LUNG', 'OVCAR4_OVARY']
    test_cell = ['TK10_KIDNEY', 'OVCAR5_OVARY', 'HOP92_LUNG', 'SKMEL2_SKIN', 'HS578T_BREAST']

    nci_data_train = nci_data[~nci_data['ccle_name'].isin(val_cell + test_cell)]

    cell2mut = pd.read_csv("./data/drug_response_data/original_cell2mut.csv", index_col=0).rename(columns={'index':'ccle_name'})
    cell2cna = pd.read_csv("./data/drug_response_data/original_cell2cna.csv", index_col=0).rename(columns={'index':'ccle_name'})
    cell2cnd = pd.read_csv("./data/drug_response_data/original_cell2cnd.csv", index_col=0).rename(columns={'index':'ccle_name'})
    drug2smi = pd.read_csv("./data/drug_response_data/DC_drug2smi.csv").iloc[:, 0:-1]

    dataset_obj = GenoDrugDataset(nci_data_train, cell2mut, drug2smi, cna=cell2cna, cnd=cell2cnd)
    collate_fn = GenoDrugCollator(genotypes=PREDIFINED_GENOTYPES)

    # Weighted sampler
    class_count = np.array([len(nci_data_train[nci_data_train['auc_label']==i]) for i in range(5)])
    weight = 1. / class_count
    samples_weight = np.array([weight[t] for t in nci_data_train['auc_label']])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))


    ##############
    # Model load
    ##############
    accelerate.utils.set_seed(42)

    batch_size = 128
    max_epochs = 2475  # 총 학습 epoch

    accelerator = Accelerator()
    device = accelerator.device

    diff_model = Diffusion(device=device, training=True, prand=0.1).to(device).to(torch.float)

    # Replace internal condition encoder to match 720 genes (ckpt expects 720)
    diff_model.model.condition_encoder = Condition_Encoder(
        num_of_genotypes=3,
        num_of_dcls=5,
        num_of_genes=720,
        gene_emb_size=128,
        device=device,
        neighbor_info=True,
        get_att=False,
    ).to(device)
    if diff_model.model.condition_encoder.gene_adj.shape[0] != 720:
        new_adj = torch.zeros((720, 720), device=device, dtype=diff_model.model.condition_encoder.gene_adj.dtype)
        h, w = diff_model.model.condition_encoder.gene_adj.shape
        new_adj[:h, :w] = diff_model.model.condition_encoder.gene_adj
        diff_model.model.condition_encoder.gene_adj = new_adj

    # ------------------------------
    # Load pretrained condition encoder (frozen) with your ckpt
    # ------------------------------
    preferred_cond_ckpts = [
        os.path.join(base_dir, "condition_encoder_pretrained.ckpt_epoch_13.ckpt"),
        os.path.join(base_dir, "condition_encoder_pretrained.ckpt_epoch_10.ckpt"),
        os.path.join(base_dir, "condition_encoder_pretrained.ckpt_epoch_2.ckpt"),
        os.path.join(base_dir, "condition_encoder_pretrained.ckpt_epoch_1.ckpt"),
        os.path.join(base_dir, "condition_encoder_pretrained.ckpt"),
    ]

    cond_ckpt_path = None
    for p in preferred_cond_ckpts:
        if os.path.exists(p):
            cond_ckpt_path = p
            break
    if cond_ckpt_path is None:
        candidates = glob.glob(os.path.join(base_dir, "condition_encoder_pretrained.ckpt_epoch_*.ckpt"))
        if candidates:
            def get_epoch(name):
                try:
                    return int(os.path.basename(name).split("_epoch_")[-1].replace(".ckpt", ""))
                except Exception:
                    return -1
            candidates = sorted(candidates, key=get_epoch)
            cond_ckpt_path = candidates[-1]
            accelerator.print(f"⚠️ Using latest found condition encoder ckpt: {cond_ckpt_path}")
        else:
            accelerator.print("⚠️ No condition encoder ckpt found; using default seed weights.")

    if cond_ckpt_path is not None:
        cond_ckpt = torch.load(cond_ckpt_path, map_location=device)
        diff_model.model.condition_encoder.load_state_dict(cond_ckpt["condition_state_dict"])
        # keep it frozen
        for _, p in diff_model.model.condition_encoder.named_parameters():
            p.requires_grad = False
        accelerator.print(f"✅ Loaded condition encoder from {cond_ckpt_path}")

    optimizer = optim.Adam([p for p in diff_model.parameters() if p.requires_grad], lr=1e-4)

    tr_loader = DataLoader(dataset_obj, batch_size=batch_size, drop_last=True, collate_fn=collate_fn, sampler=sampler)

    # prepare before loading checkpoint
    diff_model, optimizer, tr_loader = accelerator.prepare(diff_model, optimizer, tr_loader)

    total_loss = []
    start_epoch = 0

    os.makedirs("diffusion_models", exist_ok=True)

    ###########################################################################
    # 🔥 Resume: 가장 최근 checkpoint 자동 검색 및 불러오기
    ###########################################################################
    latest_ckpt = get_latest_checkpoint("diffusion_models")
    if latest_ckpt is not None:
        accelerator.print(f"🔄 Latest checkpoint found: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)

        unwrapped_model = accelerator.unwrap_model(diff_model)
        # skip old condition_encoder weights (718 genes) when resuming
        diffusion_sd = ckpt['diffusion_state_dict']
        diffusion_sd = {k: v for k, v in diffusion_sd.items() if not k.startswith("model.condition_encoder.")}
        missing, unexpected = unwrapped_model.load_state_dict(diffusion_sd, strict=False)
        if missing:
            accelerator.print(f"⚠️ Missing keys when loading diffusion ckpt (expected): {len(missing)}")
        if unexpected:
            accelerator.print(f"⚠️ Unexpected keys when loading diffusion ckpt: {unexpected}")
        optimizer.load_state_dict(ckpt['solver_state_dict'])

        total_loss = ckpt['loss_traj']
        start_epoch = int(latest_ckpt.split("_")[-1].replace(".ckpt", ""))

        accelerator.print(f"🔄 Resuming training from epoch {start_epoch}...")
    else:
        accelerator.print("🚀 No checkpoint found. Starting from epoch 0.")

    ###########################################################################


    ##############
    # Training
    ##############
    for epoch in range(start_epoch, max_epochs):
        epoch_loss = []

        for i, batch in tqdm(enumerate(tr_loader), total=len(tr_loader)):
            # pad genotype to 720 for the frozen condition encoder
            def pad_genotype(gdict, target=720, device=device):
                out = {}
                for k, v in gdict.items():
                    if v.shape[1] < target:
                        pad = torch.zeros((v.shape[0], target - v.shape[1]), device=v.device, dtype=v.dtype)
                        out[k] = torch.cat([v, pad], dim=1)
                    else:
                        out[k] = v[:, :target]
                return out

            for key in batch.keys():
                if 'genotype' in key:
                    for mut in batch[key].keys():
                        batch[key][mut] = batch[key][mut].to(device)
                    batch[key] = pad_genotype(batch[key])
                elif key in ['cell_name', 'drug_name']:
                    continue
                else:
                    batch[key] = batch[key].to(device)

            loss = diff_model(batch)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            epoch_loss.append(loss.detach().item())

        accelerator.wait_for_everyone()

        mean_loss = np.mean(epoch_loss)
        total_loss.append(mean_loss)
        print(f"Epoch: {epoch}, Loss: {mean_loss:.4f}")

        # 10 epoch마다 자동 저장
        if (epoch + 1) % 10 == 0:
            unwrapped_model = accelerator.unwrap_model(diff_model)
            save_path = f"diffusion_models/auto_save_epoch_{epoch+1}.ckpt"

            accelerator.save({
                'diffusion_state_dict': unwrapped_model.state_dict(),
                'solver_state_dict': optimizer.state_dict(),
                'loss_traj': total_loss
            }, save_path)

            print(f"✅ Checkpoint saved at {save_path}")


if __name__ == "__main__":
    main()
