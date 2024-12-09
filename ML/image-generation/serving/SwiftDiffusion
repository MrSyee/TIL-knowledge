# SwiftDiffusion: Efficient Diffusion Model Serving with Add-on Modules
- [[Paper](https://arxiv.org/abs/2407.02031)]

## Introduction

- 이미지 생성에는 Add-on (e.g. Controlnet, LoRA)을 사용하는 경우가 굉장히 많다.
    - 조사에 따르면 전체 생성 요청의 70%가 두 개의 ControlNet, 91%의 요청이 두 개의 LoRA를 사용함.
- 하지만 기존 서빙 시스템(e.g. diffuers)들은 애드온 모듈을 통합할 경우 서빙 지연시간이 길어집니다.
    - 1. 애드온 모듈은 추론 전에 GPU 메모리에 로드해야하므로 오버헤드가 발생한다(지연시간의 37%).
    - 2. 애드온 모듈은 inference시 지연 시간을 크게 연장한다(SDXL 기준 최대 5배).
- 본 논문에서는 애드온 모듈이 포함된 diffusion 모델의 추론을 원활하게 지원하는 SwiftDiffusion을 제안한다.
- SwiftDiffusion은 이미지 품질 저하 없이 이미지 생성 속도를 5배까지 높일 수 있다.

### ControlNet-as-a-Service
- 계산 그래프에 내재된 병렬성을 활용하고 컨트롤넷의 계산을 여러 GPU에 분산하여 기본 모델과 독립적으로 병렬로 실행한다.

### Efficient LoRA loading and patching
- LoRA는 모델 가중치를 기본 모델의 파라미터에 병합하는 패치를 적용해야한다.
- LoRA 패치는 일반적으로 두 단계로 이루어진다.
    - LoRA를 로컬 디스크 혹은 인메모리 태시에서 가져온다.
    - 가져온 LoRA 가중치를 기본 모델 레이어에 병합한다.
- SwiftDiffusion은 LoRA를 효율적으로 지원한다.
    - LoRA 없이 기본 모델 추론을 시작하고, 동시에 LoRA 어댑터를 가져온다.
    - 어댑터가 준비되면 기본 모델을 병합하고 나머지 작업을 완료한다.
        - 이미지 생성 초기 단계에는 LoRA가 영향을 거의 주지 않는다는 점에 기반한다.
    - 기존 LoRA 병합 작업을 최적화했다.

### Optimized UNet backbone in diffusion model
- 이미지 추론은 UNet의 노이즈 제거에 크게 의존한다(90% 이상).
- CUDA에 최적화된 연산자를 구현해 UNet 연산을 가속화한다.

- 본 논문에서는 SwiftDiffusion을 Diffusers 위에 구현하여 평균 서빙 시간을 최대 5배까지 줄이고 처리량을 최대 2배까지 개선했다.

### Main Contribution
- 대규모 text-to-image 제품에 대한 최초의 특성화 연구를 수행하고 새로운 과제를 제시한다.
- 애드온 모듈을 사용하는 이미지 생성의 효율성을 개선하기 위한 세 가지 구체적인 시스템 수준 설계를 제안한다.
- text-to-image 애플리케이션을 위한 고효율 서빙 시스템인 SwiftDiffusion을 구축하고평가한다.

## Characterization Study
- 실제 text-to-image 애플리케이션으로 특성 연구를 진행했다.
- 애플리케이션의 예시는 one-click virtual try-on과 e-commerce platform의 생성 서비스

### ControlNet Usage
- ControlNet은 총 요청 중 70% 정도에서 사용되었다. 많이 사용된다.
- 각 서비스는 50 개 미만, 혹은 100개 미만의 ControlNet을 제공하지만 그중 10% 정도의 ControlNet이 전체의 95, 98%를 차지합니다.
    - 따라서 자주 사용하는 컨트롤넷을 캐싱하는 것이 좋다. 실제로 LRU 캐시를 사용하여 컨트롤넷을 GPU 메모리에 동적으로 로드한다.
- 모델 계산 워크플로우를 분석한 결과, 기본 모델은 실행 초기에 컨트롤넷의 결과를 필요로 하지 않고, 계산 중간에 컨트롤넷을 통합한다는 사실을 발견했다.
- 이를 통해 각 컨트롤넷을 독립적인 서비스로 배포하여 성능을 가속화 할 수 있었다.

### LoRA Usage
- 보통 1, 2개의 LoRA를 사용해 스타일라이즈한다.
    - 서비스 A는 90%가 LoRA를 2개 사용, B는 74%가 1개 사용
- LoRA는 7000 ~ 7500개의 서로 다른 LoRA가 있으며 다양하게 많이 쓰여 Long tail 분포를 가진다.
- LoRA는 수가 많고 분포에 왜곡이 없어 캐시를 늘려도 오버헤드가 크게 줄어들지 않는다.
    - 모든 LoRA를 캐시하는 것은 비현실적이며 새로운 요청이 들어오면 로컬 디스크 혹은 원격 메모리 캐시에서 LoRA의 모델 가중치를 가져온다.
- 측정 결과, 원격 분산 캐시에서 800MB의 모델 가중치 2개를 가져오는데 1초 이상이 걸리며 지연시간의 최대 34%를 차지한다.
- 이미지 생성시 초기 노이즈 제거 단계에서 LoRA가 최소한의 영향을 미치는 것을 확인하였고, 이를 바탕으로 LoRA 로딩을 기본 모델 추론과 병렬화하여 로딩 오버헤드를 숨기도록 하는 방식을 고안했다.

## Design
<img width="410" alt="image" src="https://github.com/user-attachments/assets/c14a23d5-b04a-45a7-8e85-d7773be1e5f0">

- 컨트롤 넷과 기본 Diffusion 모델을 직렬 부분과 병렬 부분으로 나누어 구성했다.
-

## Reference
