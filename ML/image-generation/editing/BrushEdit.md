# BrushEdit: All-In-One Image Inpainting and Editing
- [[Paper](https://arxiv.org/pdf/2412.10316)]
- [[Project](https://liyaowei-stu.github.io/project/BrushEdit/)]
- [[Github](https://github.com/TencentARC/BrushEdit)]

## Introduction
- 이미지 편집은 inversion-based와 instruction-based 방법으로 발전하고 있다.

### Inversion-based Editing
- inversion에서 파생된 noised latent의 구조 정보를 활용하여 편집하지 않을 영역의 콘텐츠를 보존하는 동시에 편집된 영역의 latent를 조작하여 수정한다.
- 전체 이미지 구조를 효과적으로 유지한다.
- 여러 번의 diffusion sampling이 필요하여 시간이 많이 소요된다.
- inversion noise의 구조적인 특성 때문에 큰 변경(개체 추가, 개체 제거)에 약하다.

### Instruciontion-based Editing
- Source image - Instruction - target image 데이터 셋을 확보에 MLLM을 학습하여 instruction 기반으로 이미지를 편집한다.
- 데이터셋에 노이즈가 많고 신뢰하기 어려워 성능이 최적화되지 않는 경우가 많다.

### BrushEdit
- 위 두가지 한계를 해결하기 위해 MLLM의 놀라운 이미지-텍스트 이해 능력과 이미지 인페인팅 모델의 뛰어난 배경 보존 및 text-aligned foreground 생성 능력을 결합한 BrushEdit을 제안한다.
- BrushEdit은 inpainting-driven으로 동작하는 agent-based, free-form, interactive framework이다.
- 네가지 단계로 동작한다.
    - 1. 편집 카테고리 분류 (background edit, local edit, addition, remove)
    - 2. 주요 편집 대상 식별
    - 3. 편집 마스크 및 target caption 생성
    - 4. 인페인팅
- 1 ~ 3 단계는 pretrained MLLM 및 detection model을 사용한다.
- 4의 인페인팅 모델로 BrushNet을 사용한다.
- mask 기반으로 동작하는 기존의 BrushNet은 학습한 마스크가 사용자가 그린 마스크와 크게 달라서 최적의 성능을 발휘하지 못하는 문제가 있다.
    - 이 한계를 극복하기 위해 BrushNet의 Mask 데이터를 정제, 병합, 확장했다.
    - 임의의 마스크 모양을 처리할 수 있는 올인원 인페인팅 모델이 훈련되었다.

### Main Contribution
- 1. BrushNet에서 확장한 BrushEdit을 소개한다. BrushEdit은 inpainting-based image editing 방식을개척하여 제어 가능한 이미지 생성 능력을 확장한다.
- 2. 기존의 사전 학습된 MLLM 및 Vision understanding model을 통합하여 추가 학습 과정 없이 언어 이해력과 제어 가능한 이미지 생성을 크게 향상시킨다.
- 3. BrushNet을 임의의 마스크 형태도 수용할 수 있는 다용도 인페인팅 프레임워크로 확장했다.

## Preliminaries and motivation
### Image Inpainting Models
- Sampling Strategy Modification과 Dedicated Inpainting Models 두 가지 방법이 있다.
- Sampling Strategy Modification
    - 마스크된 이미지와 생성된 콘텐츠를 반복적으로 혼합하여 인페인팅을 수행한다.
    - 예시: Diffusers의 Blended Latent Diffusion(BLD)
    - 단순한 방법이지만 마스크되지 않은 영역을 보존하고 생성된 콘텐츠를 align하는 데 어려움이 있다.
        - 마스크 크기 조정으로 인해 발생하는 부정확성으로 인해 noisy latent가 적절히 혼합되지 못함
        - diffusion 모델의 마스크 경계와 마스크 되지 않은 영역에 대한 맥락적 이해가 부족
- Dedicated Inpainting Models
    - 마스크와 마스크 이미지를 추가 UNet 입력 채널로 추가하여 기본 모델을 finetuning 하여 인페인팅에 특화된 아키텍처를 만든다.
    - BLD의 성능을 능가하지만 몇 가지 도전과제가 있다.
        - Unet의 초기 컨볼루션에서 noisy latents, masked image latents, masks를 결합하게 되어 text embedding이 global하게 모든 feature에 영향을 주게 되어 더 깊은 레이어가 masked image의 디테일한 부분을 집중하기 어렵게 만든다.
        - Conditional input과 생성 작업을 동시에 처리하면 Unet의 계산 부하가 증가한다.
        - fine-tuning이 필요하므로 학습 비용과 여러 diffusion 모델에 바로 적용할 수 없는 문제가 있다.

### Image Editing Models
#### Inversion Method
- 이 방식은 다양한 inversion technique을 사용해 edit-friendly한 noisy latents를 생성한 후, 배경 영역을 보존하면서 대상 영역을 수정하는 세가지 방식을 제시한다.
    - Attention Integration
        - source와 editing diffusion branch 간의 텍스트와 이미지를 연결하는 attention map을 융합한다.
    - Target Embedding
        - target branch의 편집 정보를 임베딩하여 source diffusion branch에 통합한다.
    - Latent Integration
        - target diffusion branch에서 source diffusion branch로 노이즈가 있는 latent feature를 통해 편집 지시를 직접 주입한다.
- 이러한 방법들은 계산 효율적이고 경쟁력 있는 zero-shot, few-shot 성능을 달성하지만 편집의 다양성이 제한된다.
- inversion latents의 구조적 특성은 이미지의 큰 구조적 변화(add, remove, background replacement)를 다룰 때 종종 낮은 성능이 나온다.

#### End-to-End Method
- 다양한 ground-truth - instruction 쌍의 editing 데이터셋을 활용해 이미지 편집을 위한 end-to-end diffusion 모델을 훈련한다.
- 이 방법은 더 넓은 범위의 편집을 지원하고 inversion 방법의 속도 문제를 피하여 단일 forward pass로 편집한다.
- 데이터의 제한된 가용성으로 성능에 제약이 있다.
- 대화형 multi-turn 편집은 지원하지 않아 반복적으로 수정하거나 편집을 강화할 수 없다.

### Motivation
- 마스크 이미지를 처리하는 추가적인 분기가 있어야 더 효과적인 인페인팅 아키텍처가 된다. 이로써 백본의 추가적인 수정이나 재훈련 없이 마스크 경계와 배경을 인식할 수 있게 해준다. (Dedicated Inpainting Model 방식)
- 자유로운 형태의 인터랙티브 자연어 명령 편집 모델이 필요하다. 반복적으로 수정이 가능해야한다.

## Method
- 사전 학습된 MLLM과 dual-branch image inpainting 모델(BrushNet)을 통합하여 free-form, multi-turn interactive editing이 가능하도록 하는 BrushEdit을 제안한다.

<img width="1053" alt="image" src="https://github.com/user-attachments/assets/a8943a6a-1dda-41f1-9375-5cbaa46bfc86" />

### Editing Instructor

### Editing Conductor
- 이전 연구들은 latent blending 중 다운샘플링이 부정확성을 초래할 수 있으며 VAE 인코딩-디코딩 과정은 완전한 이미지 재구성을 저해하는 한계가 있다.
- 마스킹 되지 않은 영역의 일관된 재구성을 보장하기 위해 copy & paste 방식을 사용하지만 의미적 일관성이 부족한 출력을 만들어낸다.
- 이 연구에서는 blur된 마스크를 copy & paste 전에 사용하는 simple pixel-space approach를 제안한다.
    - 이는 마스크 경계 근처의 이미지 퀄리티(정확도)에 영향을 미칠 수 있지만 미미한 오류이며, 대신에 경계 일관성을 크게 향상시킨다.
- BrushEdit의 아키텍처는 유연한 보존 제어를 가능하게 한다.
    - 사전 학습된 확산 모델의 가중치를 수정하지 않기 때문에 어떤 fine tuning 모델과도 쉽게 통합될 수 있따.
    - BrushEdit 기능을 가중치가 고정된 확산 모델에 통합하여 마스크되지 않은 영역의 보존 규모를 제어할 수 있으며 BrushEdit이 그 수치를 조정한다.
    - 블러링 규모를 조정하고 필요해 따라 블렌딩 작업을 적용하여 보존 규모를 더욱 세밀하게 조정할 수 있다.
