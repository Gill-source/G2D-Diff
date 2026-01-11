import torch
import pandas as pd
import numpy as np
from src.g2d_diff_ce import Condition_Encoder 
from src.g2d_diff_diff import Diffusion 

# -----------------------------
# Device 설정
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}\n")

# -----------------------------
# 체크포인트 로드
# -----------------------------
ckpt_path = "diffusion_models/auto_save_epoch_290.ckpt"
print(f"Loading checkpoint: {ckpt_path}")

# torch 1.x does not support the weights_only argument, so keep the classic load signature.
try:
    ckpt = torch.load(ckpt_path, map_location=device)
except Exception as e:
    print("❌ Error loading checkpoint:", e)
    exit()

# -----------------------------
# Diffusion 모델 초기화
# -----------------------------
print("Load pretrained diffusion model ...")
model = Diffusion() 
if "diffusion_state_dict" in ckpt:
    model.load_state_dict(ckpt["diffusion_state_dict"])
    print("✅ Diffusion model loaded successfully.\n")
else:
    print("⚠️ diffusion_state_dict not found in checkpoint — using untrained weights.\n")
model.to(device)
model.eval()

# -----------------------------
# Condition Encoder 초기화
# -----------------------------
print("Load pretrained cond_encoder ...")

# 모델 가중치가 720 노드이므로, Condition_Encoder의 num_of_genes도 720으로 맞춥니다.
target_genes_param = 720 

cond_encoder = Condition_Encoder(
    num_of_genotypes=3,     
    num_of_dcls=5,          
    num_of_genes=target_genes_param,  # ✅ 720 설정
    gene_emb_size=128,
    device=device,
    neighbor_info=True
)

if "cond_state_dict" in ckpt:
    cond_encoder.load_state_dict(ckpt["cond_state_dict"])
    print("✅ Condition_Encoder loaded successfully.\n")
else:
    print("⚠️ cond_state_dict not found in checkpoint — using randomly initialized encoder.\n")

# -----------------------------------------------------------
# 🛠️ [데이터 패딩 1] 인접 행렬(Adj): 718 -> 720
# 설정값(720)과 맞추기 위해 원본 데이터를 720으로 늘립니다.
# -----------------------------------------------------------
current_adj = cond_encoder.gene_adj
required_adj_size = target_genes_param # 720

if current_adj.shape[0] != required_adj_size:
    print(f"🔧 Padding adjacency matrix: {current_adj.shape} -> ({required_adj_size}, {required_adj_size})")
    
    new_adj = torch.zeros((required_adj_size, required_adj_size), device=device, dtype=current_adj.dtype)
    
    # 원본(718x718)을 복사
    orig_h, orig_w = current_adj.shape
    new_adj[:orig_h, :orig_w] = current_adj
    
    # 교체
    cond_encoder.gene_adj = new_adj

cond_encoder.to(device)
cond_encoder.eval()
print("✅ ConditionEncoder ready.\n")

# -----------------------------
# 테스트용 데이터 불러오기
# -----------------------------
meta = pd.read_csv("./data/drug_response_data/DC_drug_response.csv")
cell_names = meta["ccle_name"].values
cell2mut = pd.read_csv("./data/drug_response_data/original_cell2mut.csv")

# 테스트할 셀 선택
cell_index = 33 
if cell_index >= len(cell2mut):
    raise IndexError(f"❌ cell_index {cell_index} out of range (max {len(cell2mut)-1})")

cell_name = cell_names[cell_index]
print(f"Using cell: {cell_name}, index = {cell_index}")

# -----------------------------
# mutation 벡터 처리
# -----------------------------
mut_vec = cell2mut.iloc[cell_index].values
mut_vec = pd.to_numeric(mut_vec, errors="coerce")

if mut_vec is None or np.isnan(mut_vec).all():
    raise ValueError("❌ Mutation vector is empty or invalid.")

mut_vec = np.nan_to_num(mut_vec, nan=0.0)
mut_vec = mut_vec.astype(float) 
mut_vec = torch.FloatTensor(mut_vec).unsqueeze(0).to(device) # (1, 718)

# -----------------------------------------------------------
# 🛠️ [데이터 패딩 2] 입력 벡터(Mut): 718 -> 720
# 모델의 최종 입력층(가중치)은 720을 기대하므로 720까지 채웁니다.
# -----------------------------------------------------------
required_input_size = 720 # 가중치 크기

if mut_vec.shape[1] != required_input_size:
    print(f"🔧 Padding mutation vector: {mut_vec.shape} -> (1, {required_input_size})")
    padded_mut = torch.zeros((1, required_input_size), device=device)
    
    # 원본(718) 복사
    padded_mut[:, :mut_vec.shape[1]] = mut_vec
    mut_vec = padded_mut

print("✅ Mutation vector loaded and converted successfully.\n")

# -----------------------------
# 인퍼런스 시뮬레이션
# -----------------------------
with torch.no_grad():
    dummy_input = {
        "genotype": {"MUT": mut_vec, "CNA": mut_vec, "CND": mut_vec},
        "class": torch.randint(0, 5, (1,)).to(device), 
    }
    
    _, cond_out, _, _ = cond_encoder(dummy_input)
    cond_out = cond_out.float()  # ensure float32 for diffusion model
    print("✅ Condition encoding complete.")

    # Diffusion.forward expects a batch dict with keys 'drug', 'class', 'genotype'.
    # 내부 Diffusion의 condition_encoder는 학습된 718 유전자 설정을 사용하므로
    # genotype 입력을 718로 잘라서 전달합니다.
    genotype_for_diffusion = {
        k: v[:, :718].contiguous() for k, v in dummy_input["genotype"].items()
    }
    batch_for_diffusion = {
        "drug": cond_out,                 # use conditioned embedding as drug input
        "class": dummy_input["class"],
        "genotype": genotype_for_diffusion,
    }

    print(f"cond_out dtype: {cond_out.dtype}, device: {cond_out.device}")
    print(f"drug dtype: {batch_for_diffusion['drug'].dtype}")
    for k,v in batch_for_diffusion['genotype'].items():
        print(f"genotype {k} dtype: {v.dtype}, shape: {v.shape}")

    pred = model(batch_for_diffusion)
    print("✅ Diffusion output shape:", pred.shape)

print("\n🎯 Test completed successfully.")
