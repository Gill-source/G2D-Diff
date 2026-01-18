# G2D-Diff Experiments

# G2D-Diff (Genotype-to-Drug Diffusion)
> A Genotype-to-Drug Diffusion Model for Generation of Tailored Anti-Cancer Small Molecules
유전 정보(Genotype: MUT/CNA/CND)로부터 약물 반응(AUC)을 예측하고, Diffusion 기반 생성 모델로 **맞춤형 항암 후보 분자(SMILES)** 를 생성하는 것을 목표로 합니다.

---

## 한눈에 보기

- **문제의식**: 환자/세포의 유전적 특성에 따라 약물 반응이 달라지므로, **유전 정보 기반 반응 예측 + 맞춤형 분자 생성**을 하나의 파이프라인으로 연결할 필요가 있습니다.
- **연구 목표**
  1) 유전 정보로부터 **Drug response(AUC)** 를 예측하는 모델 구축  
  2) **Diffusion 기반 생성 모델**로 신규 약물 구조(분자) 생성
- **구현 포인트(차별점)**
  -  3단계 구조(Condition Encoder → Diffusion → VAE-LSTM Decoder)를 유지
  - **각 모듈 독립 학습 + Condition Encoder 사전학습(pretraining)** 을 통해 latent 안정성을 높이는 방향으로 설계

---

## 전체 파이프라인

1. **Condition Encoder**
   - 입력: 유전 정보 **MUT, CNA, CND**
   - 출력: condition latent representation
2. **Regressor**
   - 입력: condition latent
   - 출력: **AUC 예측값**
3. **Diffusion Model**
   - 입력: condition latent
   - 출력: 화학적 latent (drug latent)
4. **VAE-LSTM Decoder**
   - 입력: drug latent
   - 출력: **SMILES 문자열 형태의 분자**

---

## 데이터셋 및 학습 환경

- **데이터**
  - CCLE 유전자 정보(MUT, CNA, CND)
  - Drug response AUC 데이터
- **학습 환경**
  - GPU: RTX 3090 × 2
  - Multi-GPU 분산 학습
- **학습 설정(요약)**
  - Diffusion: 300 epochs
  - Condition Encoder: 별도 pretraining 포함, 20 epochs 학습


---

## 관찰 결과
예측된 drug response(AUC)가 전체적으로 **평탄화(flattening)** 되는 경향이 확인되었습니다.  
즉, 모델 출력이 극단값(매우 민감/매우 저항)으로 충분히 벌어지지 않고 **평균 근처로 수렴(regression-to-the-mean)** 하면서, 반응 분포의 분산이 줄어드는 현상이 나타났습니다.  패딩(예: 718 → 720) 과정에서 구조 왜곡 가능성

---
