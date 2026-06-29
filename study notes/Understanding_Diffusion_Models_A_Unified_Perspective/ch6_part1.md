# [**Understanding Diffusion Models: A Unified Perspective**](../Table_of_contents.md#table-of-contents)
---
---
---

# [*VDM*](../Table_of_contents.md#table-of-contents)
---
---

## **Variational Diffusion Models**

> VDM = MHVAE + 3 restrictions

### *3 restrictions*
1. The latent dimension is exactly equal to the data dimension
2. The structure of the latent encoder at each timestep is not learned; it is pre-defined as a **linear** Gaussian
model. In other words, it is a Gaussian distribution centered around the output of the previous timestep
    - linear 조건이 반드시 있어야 한다. 
        - t-step의 $`x_t`$를 구하려 할 때, 비선형일 경우 $`0, 1, 2, ..., t`$ 까지 순차적으로 복잡한 비선형 가우시안 분포 $`q(x_i | x_{i-1})`$를 계산해 다음 단계의 $`x`$를 계산해야 한다.  
        그러나 linearity를 가정하면 수많은 step 뒤의 $`x_t`$를 구하려 할 때도, 서로 독립인 가우시안 분포들의 선형 결합은 다시 가우시안 분포가 되기 때문에,  
        순차적으로 계산할 필요 없이 t-step의 $`x_t`$를 closed-form 계산으로 바로 구할 수 있다.  (자세한 수식은 뒤에) 
3. The Gaussian parameters of the latent encoders vary over time in such a way that the distribution of
the latent at final timestep T is a standard Gaussian

제한조건 1에서 관측데이터 $`x`$와 latent $`z_{1:T}`$들의 차원 수가 같으므로 time step $`t \in [0,T]`$에 대해, 관측데이터는 $`t=0`$일 때 $`x_0`$으로 두고, latent variables들은 $`x_t,\ \text{where}\ t \in [1,T]`$로 둘 수 있게 된다. 이 조건으로 인해 HVAE의 posterior Eq. 24가 VDM에서는 

**Equation 30:**  
```math
q(x_{1:T}|x_0) = \prod_{t = 1}^{T}q(x_{t}|x_{t-1})
```

로 바뀌고,

> (HVAE와 다르게 encoder q에 $`\phi`$가 포함되어 있지 않다 => $`t`$에만 영향 받는 gaussian으로 모델링)

제한조건 2에서 forward(encoding) process의 각 단계 $`q(x_{t}|x_{t-1})`$가 신경망 모델이 아니라 미리 설정된 linear 가우시안으로 두므로, $`t`$단계의 encoder는

**Equation 31:**
```math
q(x_{t}|x_{t-1}) = \mathcal{N}(x_{t} ; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t) \mathbf{I}
```
<br>

형태가 된다.

> 평균의 계수를 $`\sqrt{\alpha_t}`$, 분산의 계수를  $`(1 - \alpha_t)`$로 특이하게 두는 이유는, 각 단계의 분산이 1로 고정(variance-preserving)되기 때문.
> - 노이즈가 추가되는 각 단계에서 이전 단계의 그림 $`x_{t-1}`$을 그대로 두고 노이즈가 계속 추가되면 분산이 계속 증가하게 된다.
>   - $`x_t`$가 서로 독립인 $`x_{t-1}`$과 $`\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`$에 대해  $`x_t = x_{t-1} + c\epsilon`$의 형태이면 $`x_{t-1}`$과 $`\epsilon`$이 서로 독립이므로 분산의 성질 $`V(aX + bY) = a^2V(X) + b^2V(Y)`$을 적용해서 $`x_t`$의 분산을 구하면, $`V(x_t) = V(x_{t-1}) + c^2 V(\epsilon) = V(x_{t-1}) + c^2`$이 되서 양수가 계속 더해져서 분산이 증가하는 것을 확인 가능  
> - 논문과 동일하게 계수를 설정하면 분산이 1로 고정된다.
>   - $`x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon`$로 두면  
$`V(x_t) = V(\sqrt{\alpha_t}x_{t-1}) + V(\sqrt{1-\alpha_t}\epsilon) = (\sqrt{\alpha_t})^2 V(x_{t-1}) + (\sqrt{1-\alpha_t})^2 V(\epsilon) = \alpha_t V(x_{t-1}) + (1-\alpha_t) V(\epsilon)`$ 이 된다.  
$`x_{t-1}`$의 분산도 $`\epsilon`$과 동일하게 1이라면, $`V(x_t) = \alpha_t (\mathbf{I}) + (1-\alpha_t) (\mathbf{I}) = (\alpha_t + 1 - \alpha_t) \mathbf{I} = \mathbf{I}`$로 분산이 $`\mathbf{I}`$, 즉 1로 고정되는 것을 확인 가능  
>  
>  
>> 평균의 계수를 $`\sqrt{\alpha_t}`$, 분산의 계수를  $`(1 - \alpha_t)`$로 두어 Variance-Preserving을 설정하는 이유 
>> - 신경망 학습의 안정화
>>   - 딥러닝 모델은 입력값의 스케일이 일정할 때 가장 빠르고 안정적으로 학습한다. 스텝 $`t`$가 1이든 1,000이든 U-Net에 들어가는 이미지의 픽셀 분산이 항상 1 근처로 유지되므로, 모델이 극단적인 숫자에 압도되지 않는다.
>> - 목표 분포(Prior $`x_T`$)로의 정확한 수렴(제한 조건 3)
>>   - 디퓨전 모델의 forward process 최종 목표는 원본 이미지를 완벽한 순수 노이즈 $`\mathcal{N}(\mathbf{0}, \mathbf{I})`$로 만드는 것. 분산이 항상 $`\mathbf{I}`$로 묶여 있어야만 마지막 스텝에서 깔끔하게 표준 정규 분포에 도달 가능.
>> - 신호 대 잡음비(SNR)의 조절 
>>   - $`\alpha_t`$ 값은 시간이 지날수록 점점 작아지도록 설정한다. 즉, $`\sqrt{\alpha_t}`$(원본 신호의 비중)는 서서히 줄어들고 $`\sqrt{1-\alpha_t}`$(노이즈의 비중)는 서서히 커지게 된다($`x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon`$). 분산을 1로 유지하면서 데이터와 노이즈의 섞이는 비율을 조절할 수 있다.

제한 조건 3을 만족하도록 forward process를 설정하므로 $`p(x_T)`$가 표준 정규분포가 된다. 제한 조건1과 3을 이용하면 HVAE의 joint distribution Eq. 23가

**Equation 32, 33:**
```math
\begin{aligned}
p(x_{0:T}) &= p(x_T)\prod_{t=1}^{T}p_{\boldsymbol{\theta}}(x_{t-1}|x_t) \\
\text{where,} \\
p(x_T) &= \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})
\end{aligned}
```
<br>

이 된다.

이 세가지 제한 조건을 만족하는 VDM을 이미지로 표현하면 아래와 같다.

![VDM](../imgs/vdm_base.png)
[이미지 출처](https://arxiv.org/abs/2208.11970)

---

## **VDM ELBO 유도 1**


**Equation 34~45:**

```math
\begin{aligned}
\log p(x)
&= \log \int p(x_{0:T}) dx_{1:T}\\
&= \log \int \frac{p(x_{0:T})q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)} dx_{1:T}\\
&= \log \mathbb{E}_{q(x_{1:T}|x_0)}\left[\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right]\\
&\geq \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right] \qquad \text{(Jensen's Inequality)}
\end{aligned}
```
```math
\begin{aligned}
&= \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)\prod_{t=1}^{T}p_{\boldsymbol{\theta}}(x_{t-1}|x_t)}{\prod_{t = 1}^{T}q(x_{t}|x_{t-1})}\right] \qquad \text{(Eq. 30 and Eq. 32)}\\
&= \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)p_{\boldsymbol{\theta}}(x_0|x_1)\prod_{t=2}^{T}p_{\boldsymbol{\theta}}(x_{t-1}|x_t)}{q(x_T|x_{T-1})\prod_{t = 1}^{T-1}q(x_{t}|x_{t-1})}\right]\\
&= \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)p_{\boldsymbol{\theta}}(x_0|x_1)\prod_{t=1}^{T-1}p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_T|x_{T-1})\prod_{t = 1}^{T-1}q(x_{t}|x_{t-1})}\right]\\
&= \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)p_{\boldsymbol{\theta}}(x_0|x_1)}{q(x_T|x_{T-1})}\right] + \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \prod_{t = 1}^{T-1}\frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})}\right]
\end{aligned}
```
```math
\begin{aligned}
&= \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log p_{\boldsymbol{\theta}}(x_0|x_1)\right] + \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)}{q(x_T|x_{T-1})}\right] + \mathbb{E}_{q(x_{1:T}|x_0)}\left[ \sum_{t=1}^{T-1} \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})}\right]\\
&= \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log p_{\boldsymbol{\theta}}(x_0|x_1)\right] + \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)}{q(x_T|x_{T-1})}\right] + \sum_{t=1}^{T-1}\mathbb{E}_{q(x_{1:T}|x_0)}\left[ \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})}\right]
\end{aligned}
```
```math
\begin{aligned}
&= \mathbb{E}_{q(x_{1}|x_0)}\left[\log p_{\boldsymbol{\theta}}(x_0|x_1)\right] + \mathbb{E}_{q(x_{T-1}, x_T|x_0)}\left[\log \frac{p(x_T)}{q(x_T|x_{T-1})}\right] + \sum_{t=1}^{T-1}\mathbb{E}_{q(x_{t-1}, x_t, x_{t+1}|x_0)}\left[\log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})}\right]\\
&= \underbrace{\mathbb{E}_{q(x_{1}|x_0)}\left[\log p_{\theta}(x_0|x_1)\right]}_\text{reconstruction term} - \underbrace{\mathbb{E}_{q(x_{T-1}|x_0)}\left[D_{\text{KL}}(q(x_T|x_{T-1}) \| p(x_T))\right]}_\text{prior matching term} - \sum_{t=1}^{T-1}\underbrace{\mathbb{E}_{q(x_{t-1}, x_{t+1}|x_0)}\left[D_{\text{KL}}(q(x_{t}|x_{t-1}) \| p_{\theta}(x_{t}|x_{t+1}))\right]}_\text{consistency term}
\end{aligned}
```
<br>

> **(Eq. 43 -> Eq. 44)**
>> ```math
>> \begin{aligned}
>> \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log p_{\boldsymbol{\theta}}(x_0|x_1)\right] 
>> &= \int \log p_{\boldsymbol{\theta}}(x_0|x_1)q(x_{1:T}|x_0)dx_{1:T} \\ 
>> &= \int \log p_{\boldsymbol{\theta}}(x_0|x_1)\bigg(\int q(x_{1:T}|x_0)dx_{2:T}\bigg) dx_{1} \\
>> &= \int \log p_{\boldsymbol{\theta}}(x_0|x_1)q(x_{1}|x_0) dx_{1} \qquad \text{(Marginalize}\ x_{2:T}\ \text{out)}\\
>> &= \mathbb{E}_{q(x_{1}|x_0)}\left[\log p_{\boldsymbol{\theta}}(x_0|x_1)\right]
>> \end{aligned}
>> ```
>
>> ```math
>> \begin{aligned}
>> \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)}{q(x_T|x_{T-1})}\right] 
>> &= \int \log \frac{p(x_T)}{q(x_T|x_{T-1})} \bigg(\int q(x_{1:T}|x_0)dx_{1:T-2}\bigg)dx_{T-1}dx_{T} \\
>> &= \int \log \frac{p(x_T)}{q(x_T|x_{T-1})} q(x_{T-1}, x_{T}|x_0)dx_{T-1}dx_{T} \qquad \text{(Marginalize}\ x_{1:T-2}\ \text{out)}\\
>> &= \mathbb{E}_{q(x_{T-1}, x_{T}|x_0)}\left[\log \frac{p(x_T)}{q(x_T|x_{T-1})}\right] 
>> \end{aligned}
>> ```
> 
>> ```math
>> \begin{aligned}
>> \mathbb{E}_{q(x_{1:T}|x_0)}\left[ \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})}\right]
>> &= \int \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})} \bigg(\int q(x_{1:T}|x_0) dx_{1:t-2} dx_{t+2:T} \bigg) dx_{t-1}dx_{t} dx_{t+1} \\
>> & = \int \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})} q(x_{t-1}, x_{t}, x_{t+1}|x_0) dx_{t-1}dx_{t} dx_{t+1} \\
>> &= \mathbb{E}_{q(x_{t-1}, x_{t}, x_{t+1}|x_0)}\left[ \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})}\right]
>> \end{aligned}
>> ```
<br>

> **(Eq. 44 -> Eq. 45)**
>> ```math
>> \begin{aligned}
>> \mathbb{E}_{q(x_{T-1}, x_{T}|x_0)}\left[\log \frac{p(x_T)}{q(x_T|x_{T-1})}\right]
>> &= \int \log \frac{p(x_T)}{q(x_T|x_{T-1})} q(x_{T-1}, x_{T}|x_0) dx_{T-1}dx_{T} \\
>> &= \int \log \frac{p(x_T)}{q(x_T|x_{T-1})} q(x_{T}|x_{T-1}, x_0)q(x_{T-1}|x_0) dx_{T-1}dx_{T} \\
>> &= \int \log \frac{p(x_T)}{q(x_T|x_{T-1})} q(x_{T}|x_{T-1})q(x_{T-1}|x_0) dx_{T-1}dx_{T} \\
>> &\text{(} x_{T}\text{에 대한 조건부 확률은 Markov property로 인해 } x_{T-1}\text{를 제외한 나머지 조건은 있어도 없어도 똑같다.)}\\
>> &= \int \bigg( \int \log \frac{p(x_T)}{q(x_T|x_{T-1})} q(x_{T}|x_{T-1}) dx_{T} \bigg) q(x_{T-1}|x_0) dx_{T-1} \\
>> &= \int \bigg( \int - \log \frac{q(x_T|x_{T-1})}{p(x_T)} q(x_{T}|x_{T-1}) dx_{T} \bigg) q(x_{T-1}|x_0) dx_{T-1} \\
>> &= - \int D_{\text{KL}}(q(x_T|x_{T-1}) \| p(x_T)) q(x_{T-1}|x_0) dx_{T-1} \\
>> &= - \mathbb{E}_{q(x_{T-1}|x_0)}\left[D_{\text{KL}}(q(x_T|x_{T-1}) \| p(x_T))\right]
>> \end{aligned}
>> ```
>
>> ```math
>> \begin{aligned}
>> \mathbb{E}_{q(x_{t-1}, x_{t}, x_{t+1}|x_0)}\left[ \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})}\right]
>> &= \int \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})} q(x_{t-1}, x_{t}, x_{t+1}|x_0) dx_{t-1}dx_{t} dx_{t+1} \\
>> &= \int \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})} q(x_{t}|x_{t-1}, x_{t+1}, x_0)q(x_{t-1}, x_{t+1}|x_0) dx_{t-1}dx_{t} dx_{t+1} \\
>> &= \int \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})} q(x_{t}|x_{t-1})q(x_{t-1}, x_{t+1}|x_0) dx_{t-1}dx_{t} dx_{t+1} \\
>> &\text{(} x_{t}\text{에 대한 조건부 확률은 Markov property로 인해 } x_{t-1}\text{를 제외한 나머지 조건은 있어도 없어도 똑같다.)}
>> \end{aligned}
>> ```
>> ```math
>> \begin{aligned}
>> &= \int \bigg(\int \log \frac{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})} q(x_{t}|x_{t-1}) dx_{t} \bigg) q(x_{t-1}, x_{t+1}|x_0) dx_{t-1}dx_{t+1} \\
>> &= \int \bigg(\int - \log \frac{q(x_{t}|x_{t-1})}{p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})} q(x_{t}|x_{t-1}) dx_{t} \bigg) q(x_{t-1}, x_{t+1}|x_0) dx_{t-1}dx_{t+1} \\
>> &= - \int D_{\text{KL}}(q(x_{t}|x_{t-1}) \| p_{\boldsymbol{\theta}}(x_{t}|x_{t+1})) q(x_{t-1}, x_{t+1}|x_0) dx_{t-1}dx_{t+1} \\
>> &= - \mathbb{E}_{q(x_{t-1}, x_{t+1}|x_0)}\left[D_{\text{KL}}(q(x_{t}|x_{t-1}) \| p_{\theta}(x_{t}|x_{t+1}))\right]
>> \end{aligned}
>> ```
<br>

### **VDM ELBO 유도 1 해석**

**Reconstruction term**  
```math
\mathbb{E}_{q(x_{1}|x_0)}\left[\log p_{\boldsymbol{\theta}}(x_0|x_1)\right]
```
<br>

VAE의 reconstruction term과 동일한 구조에 $`x`$와 $`z`$ 대신 원본 데이터 $`x_0`$와 first-step latent $`x_1`$로 교체된 형태 => 동일한 방식으로 학습

**Prior matching term**  
```math
\mathbb{E}_{q(x_{T-1}|x_0)}\left[D_{\text{KL}}(q(x_T|x_{T-1}) \| p(x_T))\right]
```
<br>

이 항은 마지막 단계 $`T`$의 latent distribution $`q(x_{T-1}, x_{T})`$이 Gaussian prior에 근사할 때 최소화되는데, 학습 가능 파라메터가 없으므로 학습하지 않는 항이다. 학습은 하지 않지만 VDM 제한조건 3( = $`p(x_T)`$ 가 표준정규분포)을 만족하도록 충분히 T를 크게해서 forward process를 설계하므로 이 항은 사실상 0이다.

**consistency term**  
```math
\mathbb{E}_{q(x_{t-1}, x_{t+1}|x_0)}\left[D_{\text{KL}}(q(x_{t}|x_{t-1}) \| p_{\theta}(x_{t}|x_{t+1}))\right]
```
<br>

이 항은 $`t`$ step의 분포 ($`x_t`$의 분포)가 forward 방향에서 구하건, backward 방향에서 구하건 일관적으로 일치하게 하도록 강제하는 항이다. 이 항의 KL div.는 VDM 모델이 $`x_{t+1}`$에서 노이즈를 제거해 만든 $`x_t`$가 $`x_{t-1}`$에서 노이즈를 추가해 만든 $`x_t`$과 얼마나 비슷한지를 측정한다. 따라서 $`p_{\theta}(x_{t}|x_{t+1})`$를 사전에 설정한 linear Gaussian(Eq. 31)인 $`q(x_{t}|x_{t-1})`$를 근사하도록 학습해야 항이 최소화된다.

이 VDM ELBO 1 해석을 그림으로 표현하면 아래와 같다.

![VDM ELBO 1](../imgs/first_derivation.png)

[이미지 출처](https://arxiv.org/abs/2208.11970)

---

## **VDM ELBO 유도 2**
[- part 2에서 계속](ch6_part2.md#vdm-elbo-유도-2)