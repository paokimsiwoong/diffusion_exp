# [**Understanding Diffusion Models: A Unified Perspective**](../Table_of_contents.md#table-of-contents)
---
---
---

# [*Three Equivalent Interpretations*](../Table_of_contents.md#table-of-contents)
---
---

VDM은 임의의 time-step t에서의 노이즈가 있는 이미지 $`x_t`$와 t 값을 이용해 원본 이미지 $`x_0`$를 예측하는 것으로 모델 학습이 가능하지만(Eq. 99), 다른 방식으로 해석을 변경해서 학습을 하는 것도 가능하다.

## **노이즈 학습하기**

먼저, $`x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\boldsymbol{\epsilon}_0\ \ \text{(Eq. 69)}`$를 $`x_0`$에 대해 정리하면

**Equation 115:**

```math
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{\sqrt{\bar{\alpha}_t}} 
```

이다.  
  
이 $`x_0`$를 $`\boldsymbol{\mu}_q(x_t, x_0)`$에 대입하면

**Equation 116~124:**

```math
\begin{aligned}
\boldsymbol{\mu}_q(x_t, x_0) &= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1 -\bar{\alpha}_{t}}\\
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)\frac{x_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{\sqrt{\bar{\alpha}_t}}}{1 -\bar{\alpha}_{t}}\\
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_{t} + (1-\alpha_t)\frac{x_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{\sqrt{\alpha_t}}}{1 -\bar{\alpha}_{t}}  \qquad \bigg(\because \frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}} = \frac{1}{\sqrt{\alpha_t}} \bigg)\\
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_{t}}{1 - \bar{\alpha}_t} + \frac{(1-\alpha_t)x_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}} - \frac{(1 - \alpha_t)\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}\\
&= \left(\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} + \frac{1-\alpha_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}\right)x_t - \frac{(1 - \alpha_t)\sqrt{1 - \bar{\alpha}_t}}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0\\
&= \left(\frac{\alpha_t(1-\bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} + \frac{1-\alpha_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}\right)x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0\\
&= \frac{\alpha_t-\bar{\alpha}_{t} + 1-\alpha_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0 \qquad \bigg(\because \prod_{i=1}^t\alpha_i \stackrel{\text{def}}{=} \bar\alpha_t \bigg)\\
&= \frac{1-\bar{\alpha}_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0\\
&= \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0
\end{aligned}
```
<br>

으로 $`x_t`$와 $`\boldsymbol{\epsilon}_0`$에 대한 식으로 정리할 수 있다.

이전에 했던 것과 동일하게 $`\boldsymbol{\mu}_{\theta}`$를 $`\boldsymbol{\mu}_{q}`$와 최대한 동일한 형태를 가지도록 모델링 하면,

**Equation 125:**

```math
\begin{aligned}
\boldsymbol{\mu}_{\boldsymbol{\theta}}(x_t, t) &= \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(x_t, t)
\end{aligned}
```
<br>

이 $`\boldsymbol{\mu}_{q}`$, $`\boldsymbol{\mu}_{\theta}`$들을 $`\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q\right\rVert_2^2\right]\ \ \text{(Eq. 92)}`$ 에 대입하면

**Equation 126~130:**

```math
\begin{aligned}
& \quad \arg\min_{\boldsymbol{\theta}} D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t)) \\
&= \arg\min_{\boldsymbol{\theta}} D_{\text{KL}}\left(\mathcal{N}\left(x_{t-1}; \boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q\left(t\right)\right) \| \mathcal{N}\left(x_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}},\boldsymbol{\Sigma}_q\left(t\right)\right)\right)\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert\frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(x_t, t) - 
\frac{1}{\sqrt{\alpha_t}}x_t + \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0\right\rVert_2^2\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0 - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(x_t, t)\right\rVert_2^2\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}(\boldsymbol{\epsilon}_0 - \hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(x_t, t))\right\rVert_2^2\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\frac{(1 - \alpha_t)^2}{(1 - \bar{\alpha}_t)\alpha_t}\left[\left\lVert\boldsymbol{\epsilon}_0 - \hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(x_t, t)\right\rVert_2^2\right]
\end{aligned}
```
<br>

Denoising matching term이  source noise $`\boldsymbol{\epsilon}_0 \sim \mathcal{N}(\boldsymbol{\epsilon}; \boldsymbol{0}, \mathbf{I})`$과 그 source noise를 학습하는 신경망 모델 $`\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(x_t, t)`$ 사이의 L2 norm 최소화로 바뀐다.
> source noise $`\boldsymbol{\epsilon}_0`$는 $`x_0`$에서 $`x_t`$로 바로 변환 할 때 추가되는 noise  
> $`\big(x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\boldsymbol{\epsilon}_0\ \ \text{(Eq. 69)}\big)`$

원래의 해석 방식으로 $`x_t`$ 원본 이미지 $`x_0`$를 예측하도록 학습하는 방식보다 두번째 해석 방식인 source noise $`\boldsymbol{\epsilon}_0`$를 예측하도록 학습하는 방식이 더 좋은 결과로 이어진다는 실험 결과를 소개하는 논문들이 있다.

---

## **Score function 학습하기**

3번째 해석 방식에서는 **Tweedie's Formula**가 쓰인다.
- 어떤 진짜 데이터에 가우시안 노이즈가 잔뜩 섞여 있을 때, 데이터 분포의 형태(기울기, Score Function, $`\nabla_{x_t} \log p(x_t)`$)만 알고 있다면, 노이즈가 낀 관측치만 보고도 원래의 진짜 데이터가 무엇이었는지 정확하게 추정(기댓값 계산 = true mean of the dist.)할 수 있다.
- Tweedie’s Formula states that the true mean of an exponential family distribution,
given samples drawn from it, can be estimated by the maximum likelihood estimate of the samples (aka empirical
mean) plus some correction term involving the score of the estimate.
    - In the case of just one observed
sample, the empirical mean is just the sample itself. It is commonly used to mitigate sample bias; if observed
samples all lie on one end of the underlying distribution, then the negative score becomes large and corrects
the naive maximum likelihood estimate of the samples towards the true mean.

가우시안 분포의 변수 $`\mathbf{z} \sim \mathcal{N}(\mathbf{z};\boldsymbol{\mu}_z, \boldsymbol{\Sigma}_z)`$에 대해 Tweedie's Formula를 적용하면
```math
\mathbb{E}\left[\boldsymbol{\mu}_z|z\right] = z + \boldsymbol{\Sigma}_z\nabla_{\boldsymbol{z}}\log p(z)
```
<br>

이 된다.

Eq. 70에서 얻은 true posterior $`q(x_t|x_0) = \mathcal{N}(x_{t} ; \sqrt{\bar{\alpha}_t}x_0, \left(1 - \bar{\alpha}_t\right)\mathbf{I})`$에서 추출한 샘플 $`x_t`$를 가지고 평균값 $`\boldsymbol{\mu}_{x_t}`$를 Tweedie's Formula로 구해보면

**Equation 131:**

```math
\begin{aligned}
\mathbb{E}\left[\boldsymbol{\mu}_{x_t}|x_t\right] = x_t + (1 - \bar{\alpha}_t)\nabla_{x_t}\log p(x_t)
\end{aligned}
```
<br>

이 된다.

분포의 true mean $`\sqrt{\bar{\alpha}_t}x_0`$을 Tweedie's Formula로 잘 근사했다고 가정하고($`\sqrt{\bar{\alpha}_t}x_0`$= 근사값 이라고 두고) 식을 전개하면

**Equation 132, 133:**

```math
\begin{aligned}
\sqrt{\bar{\alpha}_t}x_0 &= x_t + (1 - \bar{\alpha}_t)\nabla\log p(x_t)\\
\therefore x_0 &= \frac{x_t + (1 - \bar{\alpha}_t)\nabla\log p(x_t)}{\sqrt{\bar{\alpha}_t}} 
\end{aligned}
```

> 여기서 $`\nabla\log p(x_t)$는 $\nabla_{x_t}\log p(x_t)`$을 간편하게 표현한 것

이 된다.

이 $`x_0`$를 노이즈 기준 해석 전개처럼 동일하게 $`\boldsymbol{\mu}_q(x_t, x_0)`$에 대입하면

**Equation 134~142:**

```math
\begin{aligned}
\boldsymbol{\mu}_q(x_t, x_0) &= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1 -\bar{\alpha}_{t}}\\
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)\frac{x_t + (1 - \bar{\alpha}_t)\nabla\log p(x_t)}{\sqrt{\bar{\alpha}_t}}}{1 -\bar{\alpha}_{t}}\\
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_{t} + (1-\alpha_t)\frac{x_t + (1 - \bar{\alpha}_t)\nabla\log p(x_t)}{\sqrt{\alpha_t}}}{1 -\bar{\alpha}_{t}}\\
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_{t}}{1 - \bar{\alpha}_t} + \frac{(1-\alpha_t)x_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}} + \frac{(1 - \alpha_t)(1 - \bar{\alpha}_t)\nabla\log p(x_t)}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}\\
&= \left(\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} + \frac{1-\alpha_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}\right)x_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\nabla\log p(x_t)\\
&= \left(\frac{\alpha_t(1-\bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} + \frac{1-\alpha_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}\right)x_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\nabla\log p(x_t)\\
&= \frac{\alpha_t-\bar{\alpha}_{t} + 1-\alpha_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}x_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\nabla\log p(x_t)\\
&= \frac{1-\bar{\alpha}_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}x_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\nabla\log p(x_t)\\
&= \frac{1}{\sqrt{\alpha_t}}x_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\nabla\log p(x_t)
\end{aligned}
```
<br>

이 된다.  
  
이번에도 동일하게 $`\boldsymbol{\mu}_{\theta}`$를 $`\boldsymbol{\mu}_{q}`$와 최대한 동일한 형태를 가지도록 모델링 하면,

**Equation 143:**

```math
\begin{aligned}
\boldsymbol{\mu}_{\boldsymbol{\theta}}(x_t, t) &= \frac{1}{\sqrt{\alpha_t}}x_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\mathbf{s}_{\boldsymbol{\theta}}(x_t, t)
\end{aligned}
```
<br>

이 된다.

이 $`\boldsymbol{\mu}_{q}`$, $`\boldsymbol{\mu}_{\theta}`$들을 $`\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q\right\rVert_2^2\right]\ \ \text{(Eq. 92)}`$ 에 대입하면

**Equation 144~148:**

```math
\begin{aligned}
& \quad \arg\min_{\boldsymbol{\theta}} D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t)) \\
&= \arg\min_{\boldsymbol{\theta}} D_{\text{KL}}\left(\mathcal{N}\left(x_{t-1}; \boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q\left(t\right)\right) \| \mathcal{N}\left(x_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}},\boldsymbol{\Sigma}_q\left(t\right)\right)\right)\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert\frac{1}{\sqrt{\alpha_t}}x_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\mathbf{s}_{\boldsymbol{\theta}}(x_t, t) - \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\nabla\log p(x_t)\right\rVert_2^2\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\mathbf{s}_{\boldsymbol{\theta}}(x_t, t) - \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\nabla\log p(x_t)\right\rVert_2^2\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert \frac{1 - \alpha_t}{\sqrt{\alpha_t}}(\mathbf{s}_{\boldsymbol{\theta}}(x_t, t) - \nabla\log p(x_t))\right\rVert_2^2\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\frac{(1 - \alpha_t)^2}{\alpha_t}\left[\left\lVert \mathbf{s}_{\boldsymbol{\theta}}(x_t, t) - \nabla\log p(x_t)\right\rVert_2^2\right] 
\end{aligned}
```
<br>

이 되어 세번째 해석에서는 Denoising matching term이 실제 데이터 $x_t$의 분포에 대한 score 함수 $`\nabla_{x_t}\log p(x_t)`$와 이 score 함수를 학습하는 신경망 모델 $`\mathbf{s}_{\boldsymbol{\theta}}(x_t, t)`$ 사이의 L2 norm 최소화로 바뀐다.

---

## **노이즈 $`\boldsymbol{\epsilon}_0`$ 와 점수 함수 $`\nabla\log p(x_t)`$의 관계**

이 노이즈 기준 해석과 score 함수 기준 해석의 결과들을 비교하면 상수 배 정도의 차이만 있고 거의 식이 유사한 것을 확인할 수 있다. 노이즈 기준 해석에서 얻은 $`x_0`$ (Eq. 115)과 score 함수 기준 해석 전개과정에서 얻어진 $x_0$ (Eq. 133)들을 = 로 두고 풀어보면

**Equation 149~151:**

```math
\begin{aligned}
x_0 = \frac{x_t + (1 - \bar{\alpha}_t)\nabla\log p(x_t)}{\sqrt{\bar{\alpha}_t}} &= \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{\sqrt{\bar{\alpha}_t}}\\
\therefore (1 - \bar{\alpha}_t)\nabla\log p(x_t) &= -\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0\\
\nabla\log p(x_t) &= -\frac{1}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_0
\end{aligned}
```
<br>

이 되어 time-step t만 따라서 변하는 상수배 차이만 있고 완전히 동일한 형태인 것을 확인 가능하다.  
  
이는 score function은 log probability를 최대화 하는 방향(gradient of log p(x))인데, source noise는 추가되면서 원본 이미지를 손상시키므로 이 source noise의 반대 방향으로 이동하는 것은 원본 이미지로 복원되는 방향으로 생각할 수 있다. 여기서 이미지가 복원되는 방향과 log probability가 증가하는 방향 모두 true data(true mean)을 구하는 동일한 방향으로 생각할 수 있다. 결국 score 함수를 모델링하는 것은 source noise의 역방향을 모델링하는 것과 동일하다는 것을 직관과 수식으로 확인할 수 있다. 

---
---

[*Score-based Generative Models*](ch9~10.md#score-based-generative-models)
---
---