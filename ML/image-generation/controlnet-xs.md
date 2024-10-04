# ControlNet-XS: Rethinking the Control of Text-to-Image Diffusion Models as Feedback-Control Systems
- [[Paper](https://arxiv.org/abs/2312.06573)]

## Introduction
- ControlNet의 업그레이드한 ControlNet-XS를 소개한다.
- ControlNet의 파라미터를 6.5배 이상 축소해도, 이미지 품질과 제어 능력을 향상시킬 수 있다.
- 시간 측면에서 ControlNet보다 약 2배 빠르다.

### Main Contribution
- Bidirectional, high frequency and large-bandwidth communication의 중요성 입증
- 작은 사이즈의 controlling network로 픽셀 레벨 이미지 가이드, 키포인트, 포즈 가이드에 대한 제어가 가능하도록 훈련
- 20M 파라미터 네트워크로 2.6B 파라미터 네트워크인 SDXL을 컨트롤 하는 모습

## Method
### Feedback-Control System Perspective
<img width="850" alt="image" src="https://github.com/user-attachments/assets/879a1456-6370-4773-8c59-c4dedae31d51">
- Feedback-control system (= 기존 controlnet)
  - 단점: 초기 t 시점의 인코더에서 생성된 피쳐가 U-net을 거쳐 t+1에만 제어 신호를 수신할 수 있다.
  - 이미 35개의 생성 블록을 거친 상태이기 때문에 초기 생성된 특징(그림의 빨간 네모 부분)을 인식하는 초기 제어 신호가 없다.
- High-frequency communication
  - 초기 생성된 특징을 한번의 생성 블록 후에 바로 제어 신호를 수신할 수 있다.
  - 더 많은 주기로 제어 신호를 주고 받기 때문에 고주파 통신이라고 부른다.
- Large-bandwidth
  - frequency 뿐 아니라 bandwidth도 중요하다.
  - 기존 방식에서는 64x64x4=524K bit 만이 다음 단계로 이동한다.
  - 새로 제안한 방법에서는 212M bit (약 404배 이상)이 이동한다.
- 결론적으로 기존 방법은 생성 프로세스에서 적은 입력을 받기 때문에 적절한 제어 신호를 계산하는 것이 어렵다.

<img width="815" alt="image" src="https://github.com/user-attachments/assets/ce03da34-1257-4015-8366-862e8b8495fb">
- 최종적으로 Type B를 ControlNet-XS라고 명명함

### ControlNet
- Control model은 UNet의 인코더를 복사한다.
- Control encoder는 control signal과 중간 생성된 noisy image를 입력받는다.
- 그리고 생성 프로세서의 다른 decoder block에 공급되는 제어 신호를 출력한다.
- 생성 모델로의 연결은 zero-convolution으로 초기화되어 훈련 초기에 UNet의 생성 기능을 저하시키지 않도록 한다.
- 학습 중에 control encoder는 생성 프로세스에 유용한 제어 신호를 제공하는 방법을 학습 할 수 있다.
- control system 관점에서 제어 모델은 두가지 작업을 동시에 수행해야한다.
  - 피드백 신호를 생성 프로세스에 유용하도록 처리
  - 제어 신호가 생성 모델에 수신될 때까지 생성 프로세스가 무엇을 할 것인지 예측
- 논문에서 제안한 모델은 두번째 작업을 해결하여 제어 모델은 첫번째 작업에 집중할 수 있도록 한다.

### ControlNet-XS
- 핵심 아이디어는 두 인코더가 높은 주기로 상호작용한다는 것이다. 이를 위해 세가지 변형을 소개한다.
- b, c, d 중 c, d가 설계 관점에서 좋다. b는 제어 루프가 많지만, 여전히 루프 내에서 uncontrolled processing을 수행한다.
  - (*)생성 인코더에 제어 신호가 들어가지 않는다는 말로 보인다.
- 실험 적으로 성능이 b < c = d을 검증했다. c가 d보다 파라미터 수가 적기 때문에 c (type B)를 최종 아키텍쳐로 선택했다.
- 제어 모델에서 계산된 feature는 zero-convolution으로 처리되어 생성 모델에 추가된다. 생성 블록에서 생성된 특정도 제어 블록에 추가될 수 있다.
- 새로운 설계를 통해 각 제어 계층의 채널 수를 일관되게 변경함으로써 제어 네트워크의 크기를 대폭 줄일 수 있다. (361M -> 1.7M)
  - 구조 변경 없이 단순히 네트워크 크기를 줄인 버전인 ControlNet-light 버전의 경우 성능이 더 떨어지는 것으로 나타났다.


## Reference
- https://vislearn.github.io/ControlNet-XS/
- diffusers patch: https://github.com/huggingface/diffusers/releases/tag/v0.28.0
- Benchmark: https://github.com/UmerHA/controlnet-xs-benchmark/blob/main/Speed%20Benchmark.ipynb
