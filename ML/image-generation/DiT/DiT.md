# Scalable Diffusion Models with Transformers
- [[Paper](https://arxiv.org/pdf/2212.09748)]

## Introduction
- Transformer 구조 기반의 새로운 Diffusion 모델을 제안한다. 본 논문에서는 Diffusion Trnasformers(DiTs)라고 부른다.
- U-Net의 유도 편향(inductive bias)가 Diffusion 모델 성능에 중요하지 않으며, transformers로 쉽게 대체할 수 있음을 보인다.
- 보다 구체적으로, 네트워크 복잡성 vs 샘플 퀄리티와 관련하여 transformers의 확장에 대해 연구한다. 네트워크 복잡성(Gflops)과 샘플 품질(FID) 사이에는 강한 상관관계가 있으며 DiTs가 확장 가능한 Diffusion 모델 아키텍처라는 것을 보여준다.

<img width="1007" alt="image" src="https://github.com/user-attachments/assets/d5803608-b4df-4b43-97cb-ab5113553d6d">

## Diffusion Transformers
### Preliminaries
- Diffusion formulation
- Classifier-free guidance
- Latent diffusion models
- Architecture complexity
  - 일반적으로 파라미터 수는 모델 아키텍처 복잡성을 평가할 때 관행적으로 사용한다.
  - 그러나 이미지 생성에 대해서는 이미지 해상도 등을 고려하지 않기 때문에 적절하지 않을 수 있다.
  - 본 논문에서는 **Gflops** 지표를 통해 복잡성을 측정한다.

### Diffusion Transformer Design Space
- ViT 기반으로 DiT를 구성했다.

#### Patchify
<img width="656" alt="image" src="https://github.com/user-attachments/assets/0944b509-3e8d-40ca-b221-7019a1dd07f9">

- DiT의 첫번째 레이어는 patchify layer이다.
- spartial input을 T tokens의 sequence로 변환한다.

#### DiT block design
서로 다른 conditional inputs를 처리하기 위해 four variants of transformer blocks를 만들었다.

- In-context conditioning
- Cross-attention block
- Adaptive layer norm(adaLN) block
- adaLN-Zero block

#### Model size
모델 사이즈에 따라 DiT-S, DiT-B, DiT-L, DiT-XL을 제안한다. 모델 사이즈는 Layer 개수 N, Hidden size d, attention head 수에 따라 달라진다.

#### Transformer decoder
- 마지막 DiT block을 지나면 image token을 output noise prediction과 output diagonal covariance prediction으로 decode 해주어야한다.
- 두 output의 shape은 DiT block의 입력 shape과 같다.