# InsightEdit: Towards Better Instruction Following for Image Editing
- [[Paper](https://arxiv.org/pdf/2411.17323)]
- [[Project](https://poppyxu.github.io/InsightEdit_web/)]

## Introduction
- InstructPix2Pix, Instruct-Diffusion, SmartEdit과 같이 instrucion-based end-to-end image editing 연구가 계속 되고 있다.
  - 이 방법은 MLLM(Multimodal LLM)을 사용하여 명령어를 이해함으로써 복잡한 editing 작업에 대처하도록 한다.

- 그러나 두가지 챌린지가 남아 있다.

1. 고품질 데이터의 부족

- 현재 데이터셋은 low resolution과 poor visual quality, background consistency, overly simplistic 문제가 있다.
- 이 한계 때문에 복잡한 명령의 이해와 high-fidelity target image generation 능력이 떨어진다.


- 데이터셋 문제를 해결하기 위해 complex instruction과 strong background consistency를 갖춘 editing data 쌍을 생성하는 자동화된 데이터 구성 파이프라인을 제안한다.
  - MLLM의 인식 기능을 활용하여 고해상도 이미지에서 상세한 오브젝트 정보를 추출하고
  - advanced mask-based editing 모델을 활용해 사실적이고 제어 가능한 편집 데이터를 생성했다.
  - 2,500,000 개 이상의 데이터셋인 AdvancedEdit을 소개한다.

<img width="855" alt="image" src="https://github.com/user-attachments/assets/9e178b70-bb11-4a8f-92f6-e00cc1612d6a" />


2. 풍부한 이미지 condition의 부족

- 현재 방식은 CLIP 텍스트 인코더를 사용하여 condition을 제공하지만, 명령어를 이해하기에 한계가 있는 경우가 많다.
- 이를 해결하기 위해 MLLM을 활용하지만, 여전히 이미지의 풍부한 시각적 의미(visual semantics)를 포착하는 데는 소홀하다.
- 그래서 복잡한 명령어를 할 수 없고, 배경 일관성을 유지하는 데 약하다.


- Rich image condition 부족 문제를 해결하기 위해 two-stream bridging 메커니즘을 사용하여 높은 수준의 textual 정보와 풍부한 visual 정보를 diffusion 모델의 denoising process에 통합한다.


- 두 가지 방법을 활용한 모델은 Reason-Edit과 AdvancedEdit-Eval 평가에서 모두 SOTA를 달성하였다.

### Main contribution
1. Instruction-based image editing 모델의 학습을 용이하게 하기 위해 이미지 편집을 위한 자동화된 데이터 구축 파이프라인을 제안한다.
2. 복잡한 명령어와 우수한 배경 일관성을 갖춘 대규모 이미지 편집 데이터셋인 AdvancedEdit 데이터셋을 소개한다.
  - 복잡한 명령어를 처리하는 모델의 능력을 평가하기 위해 AdvancedEditEval 데이터셋도 소개한다.
3. two-stream bridging 메카니즘을 활용해 MLLM을 통해 얻은 textual, image feature를 둘다 이미지 편집에 활용하는 모델인 InsightEdit 모델을 소개한다.


## Related Work
- 최근 image editing의 발전은 mask-based와 mask-free 둘로 나눌 수 있다.

### Mask-based
- Mask-based 방식은 다시 그리고 싶은 부분을 세밀하게 제어할 수 있어 안정적이고 시각적 품질이 높은 결과물을 얻을 수 있다.
- 하지만 추가 마스크 정보가 필요하고, 마스크 모양과 특정 편집 작업에 민감하다.

- Blended Latent Diffusion
    - 노이즈 제거 프로세스 중에 특정 영역을 대상으로 다시 그릴 수 있도록 확장한 모델
- BrushNet
    - 이중 Unet 구조를 사용하여 마스크된 영역에서 특징 추출을 향상시키기 위한 추가 분기를 사용
- Power-Paint
    - 다양한 편집 작업을 분류하는 토큰을 학습하여 다양한 이미지 편집 작업의 고유한 특성에 초첨을 맞춤
- Imagen editor and editbench: Advancing and evaluating text-guided image inpainting
- Smartbrush: Text and shape guided object inpainting
with diffusion model
- Text-guided subject-driven image inpainting with diffusion models

### Mask-free
- Mask-free 방법은 명시적인 마스크에 대한 의존성을 줄이고 보다 유연하고 직관적인 이미지 편집을 가능하게 한다.
- 마스크 기반 방식에 비해 이미지 품질이 떨어지는 경우가 많다.

- InstructPix2Pix
    - GPT3 + Prompt2Prompt 방법론을 활용해 이미지 편집 데이터셋을 구축한다.
- InstructDiffusion
    - InstructPix2Pix기반의 네트워크 디자인으로 다양한 비전 태스크를 통합하는 것을 목표로함
- MGIE
    - MLLM으로 표현적 명령어를 생성하는 방법을 학습하여 instruction-based 이미지 편집을 향상시킴
- SmartEdit
    - 텍스트 임베딩을 생성하는 데 MLLM을 활용


## Dataset Construction
### Automated Pipeline
- high-fidelity, fine-grained image editing pair를 생성하는데 집중한 파이프라인을 구축했다.
- image editing을 세가지 카테고리로 나누었다.
    - removal
    - addition
    - replacement

<img width="1005" alt="image" src="https://github.com/user-attachments/assets/3b625615-acac-4c97-a96f-67905943c1ac" />

#### Step 1: Caption & Object Extraction
- 이미지의 내용을 효과적으로 전달할 수 있는 글로벌 캡션을 생성하기 위해 MLLM의 고급 이해 기능을 활용한다.
- LLM으로 오브젝트를 JSON 리스트로 만들고 물리적 의미가 있는 오브젝트를 식별한다.
- 각 객체는 간단한 캡션과 자세한 설명을 정의한다.

#### Step 2: Mask Generation
- GroundedSAM으로 각 오브젝트에 대한 로컬 마스크를 추출하고 사전 정의된 임계값에 따라 신뢰도가 낮은 마스크를 피러링하여 추가 처리를 위한 오브젝트와 마스크 쌍을 얻는다.

#### Step 3: Editing Pair Construction
- 마스크 기반 이미지 편집 모델은 이미지 생성기능이 뛰어나 특정 작업에 대한 생성 프로세스를 더 잘 제어할 수 있습니다.
- 최신 마스크 기반 방법을 사용해세밀하고 제어 가능한 이미지 편집 쌍을 생성하였다.
    - BrushNet, PowerPaint
- 이미지 편집 생성 작업을 제거, 추가, 대체의 세가지 유형으로 분류했다.
- 제거
    - 원본 이미지에서 마스크 부위를 mask-based 편집 모델로 제거함. Instruction template은 (remove the [object]).
- 추가
    - 제거에서 target과 원본을 뒤바꾸면 추가 데이터셋이 된다. Instruction template은 (add the [object]).
- 대체
    - MLLM의 기능으로 장면에 대한 대체 오브젝트를 제안하고 mask-based 모델로 대체 생성한다. Instruction template은 (replace the [source object] with the [target object]).

#### Step 4: Instruction Recaptioning
- Instruction의 표현을 다양화하기 위해 Simple과 Advanced 버전으로 재구성한다.
- Simple
    - 동의어 대체 및 작업 템플릿 수정을 통해 생성된다.
    - replace the [acoustic guitar] with [white guitar] -> Swap the [acoustic guitar] for the [white guitar]
- Advanced
    - 단순한 개체 설명을 1단계에서 준비한 상세한 설명으로 대체하고 다른 작업 템플릿을 적용한다.
    - 추론 능력을 키울 수 있는 지시를 다시 제시합니다.
        - 예를 들어 What is the man holding in his hands? Please change the color of this object to white. 과 같은 지시를 만듬.

#### Step 5: Quality Evaluation
- [VIEScore](https://tiger-ai-lab.github.io/VIEScore/)로 평가하였다. 이 평가 또한 VLM을 이용해서 한다.
- 두가지 요소
    - Semantic consistency
        - instruction을 잘 따르는지와 이미지가 과도하게 편집되었는지
    - Perceptual quality
        - 아티팩트의 흐릿함 등 이미지의 충실도를 평가하는 지각적 품질

### AdvancedEdit Dataset
- High-quality real-world photographic image dataset인 Pexels를 소스 데이터로 선택했다.
    - 평균 해상도가 2K 이상
- 각 이미지에 여러 오브젝트가 포함되어 있고 다양한 명령어 작업과 연괸되어 있다.
- 데이터셋은 2,536,674 개의 편집 쌍으로 구성되었다.
- 간단한 instruction이 포함된 이미지 편집 쌍을 SimpleEdit
- 복잡한 instruction이 포함된 이미지 편집 쌍을 AdvancedEdit
- 제거, 추가, 교체 등 다양한 작업을 다루는 300개의 선별된 이미지 쌍으로 구성된 AdvancedEdit-Eval

## Method
### Overview
- Comprehension module
    - MLLM을 통해 image editing task를 이해한다.
- Bridging module
    - text feature와 image feature를 diffusion의 denoising process에 통합한다.
- Generation module
    - editing guidance를 받아 diffusion 모델을 통해 target 이미지를 생성한다.

<img width="1018" alt="image" src="https://github.com/user-attachments/assets/eb51ec0c-7ae9-4e5d-a9e1-e6e7ca3ec32e" />

### Comprehension module
- LLaVA-7B를 사용했다.
- 이미지를 vision encoder에 통과시켜 얻은 이미지 임베딩 v
- instruction을 토크나이즈해 구한 텍스트 임베딩 c
- 특수한 [MM] 토큰을 도입했다. LLM을 통해 텍스트와 이미지 정보의 multi-modality 이해를 의미한다.
- v, c, 이전 MM들이 입력되었을때 다음 MM을 예측하는 문제를 학습시킴.

### Bridging Module
- textual feature는 high-level 정보를 제공하는 반면 image features는 더 detail한 조건을 제공한다.
- Bridging model은 textual과 image feature 모두를 조건으로 포괄적으로 통합하기 위해 dual-stream condition alignment 방법을 설계한다.

#### Textual Branch via Q-Former & BIM
- Textual branch는 SmartEdit과 유사하게 text-aligned Q-Former로부터 시작한다.
- Q-Former는 우선적으로 original clip text embedding space와 align을 맞춘다.
- 그런 다음 학습 가능한 토큰 q가 입력으로 주어지면, cross-attention을 통해 [MM] 토큰 hidden state h에서 추론된 텍스트 정보(q')를 추가로 추출한다.
- 추가로 SmartEdit을 참고해, 이미지 입력과 MLLM output 사이의 양방향 정보 교환을 가능하게 하는 BIM module을 사용한다.
- f_txt와 v_txt는 BIM 모듈에서 얻은 상호 작용된 특징이며, f_txt는 text condition으로 사용되고, v_txt는 Unet 입력에 추가되는 textual-aware vision feature이다.

#### Image Branch via IAA
- Image branch에는 Image Alignment Adapter(IAA) 모듈을 제안한다.
- IAA 모듈은 MLLM로 추론된 이미지 정보를 활성화한다.
- IAA 모듈은 mapper, linear layer, layer normalization으로 구성된다.
- [MM] token hidden states h를 임베딩하는 Mapper를 타겟 이미지를 CLIP 임베딩한 feature를 만들어내도록 학습한다.
- 이렇게 학습한 IAA의 결과물인 이미지 feature를 Unet의 Cross-attention의 입력으로 사용한다.

<img width="412" alt="image" src="https://github.com/user-attachments/assets/402173da-56dc-42ae-ac0d-8e6bbc051f61" />

### Generation Module
- IP-adapter에 영감을 받아, textual과 이미지 feature를 통합하기 위해 decoupled cross-attention 메카니즘을 적용했다.


## Results
### Experiment Settings
#### Implementation Details
- 학습 데이터로 CC11M, InstrctPix2Pix, MagicBrush, COCO, RefCOCO, GRefCOCO, COCOStuff, LISA, ReasonEdit 그리고 AdvancedEdit을 사용했다.
- 자원의 제한으로 AdvancedEdit의 202,822 editing pair를 실험에 사용했다.
- InsightEdit은 3 stage로 학습했다. (디테일한 정보는 appendix에 있다. *그런데 appendix가 없음)
- H100 GPU 8장을 사용했다.
- MLLM으로 GPT-4o를 사용했다.
- LLM으로 Qwen2를 사용했다.

#### Metrics
- CLIPScore, PSNR, SSIM, LPIPS를 측정했다.
    - PSNR, SSIM, LPIPS는 background consistency를 측정하기 위함.
- VIEScore가 인간 선호도에 잘 부합하여 설득력있는 척도가 될 수 있음을 강조한다.
    - instruction-following과 background consistency 측면의 성능을 측정할 수 있다.

### Comparison with State-of-the-art

#### Quantitative results
<img width="715" alt="image" src="https://github.com/user-attachments/assets/ff186572-d003-41be-bd34-6b57e3bde3b7" />

#### Qualitative comparison

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/8ddf46b5-399d-4f50-b9e6-c50ab62b99d9" />
