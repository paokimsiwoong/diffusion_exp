"""
출처: https://nn.labml.ai/diffusion/ddpm/index.html
"""
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from utils import gather
# gather는 1~T의 값을 모두 가지는 alpha, beta등의 텐서에서 torch.gather 함수로 배치에서 지정된 t step의 값들만 뽑아낸 뒤(1차원)
# 이미지 텐서와 동일한 dimension(batch, ch, height, width의 4차원)을 가지도록 변경해주는 util 함수


class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $T$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model

        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        # DDPM 원논문과 동일하게 β_1은 10^-4, β_T 는 0.02로 설정

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta

        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # torch.cumprod는 (x_0, x_1, x_2, ..., x_i, ...)인 벡터를 받으면
        # (x_0, x_0*x_1, x_0*x_1*x_2, ..., x_0*x_1*....*x_i, ...)를 반환

        # $T$
        self.n_steps = n_steps

        # $\sigma^2 = \beta$
        self.sigma2 = self.beta
        # 이 σ^2은 q(x_t | x_{t-1})의 분산 
            # q(x_t | x_{t-1})는 평균이 √α_t * x_{t-1}이고 분산이 (1-α_t)I 인 isotropic Gaussian
            # 1-α_t = β_t

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)

        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps
        # x ~ q(x_t|x_0) 이고 q(x_t|x_0)는 평균이 μ, 분산이 σ^2인 isotropic Gaussian
        # ==> x = μ + σ * ε

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{1 - \alpha_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t)

        # gather $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)

        # $\frac{1 - \alpha_t}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5

        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{1 - \alpha_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # $\sigma^2$
        var = gather(self.sigma2, t)
        # p_θ(x_{t-1} | x_t)는 q(x_t | x_{t-1})가 아니라
        # q(x_{t-1} | x_t, x_0)를 모델링하므로 
            # 분산은 t-step 마다 fixed value이므로 모델링할 때 분산 값은 동일하게 설정
        # q(x_t | x_{t-1})의 분산(β_t == 1-α_t)가 아니라
        # q(x_{t-1} | x_t, x_0)의 분산 (1-α_t) * (1-\bar\alpha_{t-1}) / (1-\bar\alpha_t)를 써야 하지만
        # 두 분산은 상수배 차이고 스케일 차이가 거의 없어서 수식적으로 단순한 β_t를 써도 생성된 이미지의 품질에 차이가 없다
            # 분산 값은 각 t step에서 추가되는 노이즈의 폭(최대치)만 결정하고 β값 자체가 10^-4 ~ 0.02 사이 값으로 매우 작아서 
            # 두 분산을 바꾸어써도 이미지에 추가되는 노이즈의 스케일 크기 차이가 거의 없다.
            # DDPM 원 논문을 보면 두 분산 두가지 다 적용해서 실험 후 차이가 없음을 확인함
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # q(x_t | x_{t-1})의 분산과 q(x_{t-1} | x_t, x_0)의 분산은
        # 역방향 과정(복원 과정)에서 가질 수 있는 분산의 양극단(상한과 하한)이다.
        # 상한
            # β_t는 정방향 과정(노이즈 추가 과정)에서 노이즈를 더할 때 사용한 분산값
            # q(x_t | x_{t-1})의 분산은 x_0를 모르는 상태에서(x_0가 어떤 사진 이미지인지 단순한 완전 노이즈인지 모름)
            # 불확실성 최대 -> 복원 최종 결과 x_0를 모르므로 역방향으로 돌아갈 때, 정방향과 같은 크기의 노이즈를 그대로 사용
            # ==> 모델이 가질 수 있는 가장 큰 분산 값(상한)
        # 하한
            # 원본 이미지(x_0)가 무엇인지 완벽하게 알고 있으므로 x_t에서 x_{t-1}로 복원할 때 최종 복원 결과를 알고 있어
            # 어디로 복원해야 하는지 불확실성이 가장 적다. ==> 이 때의 분산이 모델이 가질 수 있는 가장 작은 분산 값(하한)
                # (1-\bar\alpha_{t-1}) / (1-\bar\alpha_t) 에서 \bar\alpha_t는 
                # \bar\alpha_{t-1} * \alpha_t로 \alpha_{t-1} 보다 항상 작다. 
                # ==> (1-\bar\alpha_{t-1}) / (1-\bar\alpha_t)는 항상 1보다 작다
                # ==> q(x_{t-1} | x_t, x_0)의 분산은 q(x_t | x_{t-1})의 분산 보다 항상 작다.
        # 그러나 T = 1000으로 각 t step 간 추가되는 노이즈의 분산 값 차이, \bar\alpha_t와 \bar\alpha_{t-1} 사이 차이가 매우 작아져
        # 사실상 두 분산의 값은 거의 같아진다.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)

        # Sample
        return mean + (var ** .5) * eps
        # x = μ + σ * ε

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$

        원래의 loss의 summation 부분 ∑_{t=2}^T (L_{t-1}) 항은 999개의 L_{t-1}를 계산해야 해서 매우 큰 연산이 필요하다.
        여기서 ∑_{t=2}^T (L_{t-1})를 (T-1) * ∑_{t=2}^T (1/T-1)*(L_{t-1}) 형태로 바꾸면 (1/T-1)이 U{2,T} 분포의 prob. mass 값이 된다.
        이렇게 보면 summation을 평균으로 변경 가능해진다.
        ===> 평균으로 변경하고 나면 배치마다 몬테카를로 샘플링으로 추출된 임의의 t에 대해서만 L_{t-1}을 계산한다.
        
        추가로 DDPM 원 논문에서는 근사를 통해 L_0도 L_{t-1}과 동일한 MSE|source noise - pred. noise| 형태로 변경해 
        t = 2 ~ T 범위를 t = 1 ~ T로 늘려 L_0와 L_{t-1}을 하나의 L_simple로 통합해 버린다.

        + 원래의 loss 항에 있던 가중치(상수 항)을 무시하는데 이렇게 무시를 하면 작은 t step 부근의 L 값을 작게 만든다.
            ==> 전체 loss에서 큰 t step 부근의 loss 비중이 커지므로 
                모델이 더 어려운 큰 t step 부근의 노이즈 복원 학습에 더 집중하도록 만든다. 
                이렇게 하면 상수 항을 무시하지 않고 학습한 모델보다 샘플 품질이 더 좋다.
        (∵ 상수 항의 분모에 있는 1 - \bar\alpha_{t-1}은 t가 작을 때 \bar\alpha_{t-1} ~ 1 이므로 0에 가까운 값이 된다
                (α_t = 1 - β_t 이고 t가 작을 때 β ~ 0 & \bar\alpha_t = ∏_1^t α_i로 1 보다 작은 값이 계속 곱해지므로 t가 클수록 \bar\alpha_t이 1에서 0 방향으로 작아진다)
            => 가중치(상수 항) 값 ~ 무한대
            ==> t가 작을 때의 가중치가 엄청나게 커져서 모델이 이미 복원이 거의 완료된 상황에 원본과 픽셀 값 1~2 차이 나는 것을 
            t가 클 때 거의 노이즈인 이미지에서 원본 이미지의 윤곽을 잡는 상황에서의 픽셀이 크게 차이나는 것보다 수천배 더 큰 에러로 받아들인다.
            ===> 이 항을 1로 바꾸어서 t 값에 따른 크기 변화를 지워버리면 모델이 거의 다 완성된 이미지에서 눈에 보이지 않는 차이를 고치는데 집중하기보다
            t가 클때의 거의 노이즈인 이미지에서 전체적인 구조와 형태를 스케치하는 복원 과정에 더 집중해서 학습하게 되어 좋은 품질의 이미지 샘플을 생성하게 된다.)

        

        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return F.mse_loss(noise, eps_theta)