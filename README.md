# diffusion_exp
diffusion 계열 논문 리뷰 및 모델 실험

# *Table of contents*
>## 1. [README](#readme)
>>### 1. [개요](#개요)
>>### 2. [사용법](#사용법)
>## 2. [Understanding Diffusion Models: A Unified Perspective](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch1~3.md)
>>### 1. [Generative Models](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch1~3.md#generative-models)
>>### 2. [Background](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch1~3.md#background)
>>### 3. [ELBO](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch1~3.md#elbo)
>>### 4. [VAE](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch4~5.md#vae)
>>### 5. [HVAE](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch4~5.md#hvae)
>>### 6. [VDM](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch6_part1.md#vdm)
>>### 7. [Learning Diffusion Noise Parameters](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch7.md#learning-diffusion-noise-parameters)
>>### 8. [Three Equivalent Interpretations](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch8.md#three-equivalent-interpretations)
>>### 9. [Score-based Generative Models](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch9~10.md#score-based-generative-models)
>>### 10. [Guidance](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch9~10.md#guidance)

[관련 논문 리뷰 노트 ipynb](./study%20notes/diffusion%20model%20이론%20공부.ipynb)

---
---
---
---

# [**README**](#table-of-contents)
---
---
---

## [**개요**](#1-개요)

Diffusion 계열 모델 논문들을 공부하고 내용을 정리한 필기 노트와 해당 모델들을 구현하고 실험하는 코드들을 보관하는 저장소.

1. DDPM 모델  
    - [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)  
        Variational diffusion model(VDM)이 어떠한 이론적 배경을 가지고 있는지 Variational auto encoder(VAE) -> Hierarchical VAE -> Markovian HVAE -> VDM 순으로 설명하는 논문
    - [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  
        DDPM 모델 원 논문. diffusion 모델의 학습 목표(loss)를 개선해서 안정적으로 학습이 되면서 동시에 GAN등의 다른 계열 이미지 생성 모델에 비슷하거나 더 고품질의 이미지를 생성할 수 있음을 보인 논문.
    - [labml.ai](https://nn.labml.ai/diffusion/ddpm/index.html)  
        DDPM 모델 코드 레퍼런스. 여기서는 원코드에서 실험을 관리하는 labml 라이브러리대신 wandb 라이브러리를 사용하도록 코드를 변경. 

## [**사용법**](#2-사용법)

1. 이 저장소를 원하는 경로에서 clone
    - git clone https://github.com/paokimsiwoong/diffusion_exp.git
2. 저장소 root 경로에서 uv 패키지 매니저로 프로젝트 생성
    - uv init
3. 가상환경 생성
    - uv venv
4. 저장소에 포함된 pyproject.toml 파일을 참고해 필요 라이브러리를 설치
    - uv sync
5. 각 모델 별 폴더 안의 experiment.py 파일을 uv로 실행
    - uv run ddpm/experiment.py
        - 실행하면 모델 설정을 담은 yaml 파일의 경로를 입력하는 창이 나온다. 예시 파일을 참조해 yaml 파일을 작성 후 해당 파일 경로를 입력한다.
        - 다음에는 실험 이름을 입력하면 실험이 시작.
            - 이 이름으로 폴더를 생성해 그 안에 실험 중 생성되는 파일들을 저장하고, wandb에 지정한 이름으로 실험(run)을 생성해 실험 결과들을 기록한다.

---
---
---

# [**Understanding Diffusion Models: A Unified Perspective**](./study%20notes/Understanding_Diffusion_Models_A_Unified_Perspective/ch1~3.md)
---
---
---