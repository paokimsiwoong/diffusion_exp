# [**Understanding Diffusion Models: A Unified Perspective**](../Table_of_contents.md#table-of-contents)
---
---
---

# [*VDM*](../Table_of_contents.md#table-of-contents)
---
---

### [**$q(x_{t-1}|x_t, x_0)$ == tractable**](ch6_part2.md#-tractable---part-3에서-계속)

ELBO 유도 2의 3가지 항에서 학습을 진행할 때 가장 많은 비용이 들고 영향을 주는 부분은 summation이 있는 3번째 항. Markovian HVAE의 경우에는 forward process의 T개의 encoder $q_{\phi}$들을 decoder $p_{\theta}$와 같이 동시에 학습하며 $D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t))$을 최소화해야 하기 때문에 매우 어렵다. 반면에 VDM의 경우 제한조건에 의해서 encoder $q_{\phi}$들을 학습하지 않고, 미리 정해두므로 $q(x_{t-1}|x_t, x_0)$가 tractable이 된다.

Bayes rule을 이용하면 $q(x_{t-1}|x_t, x_0)$ 은

```math
q(x_{t-1}|x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0)q(x_{t-1}|x_0)}{q(x_{t}|x_0)}
```
<br>

로 표현이 된다.  

> $q(x_{t-1}|x_t, x_0)$항은 $x_t$와 $x_0$이 주어졌을 때, 어떤 $x_{t-1}$이 가장 그럴듯한(잘복원된)지 알려주는 확률분포이고, Bayes rule을 적용한 우변에서 $q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$은 $x_{t-1}$에서 다시 $x_t$로 가는 것이 얼마나 쉬울까 알려주는 확률분포, $q(x_{t-1}|x_0)$은 주어진 $x_0$에서 $x_{t-1}$로 가는 것이 얼마나 쉬울까 알려주는 확률분포

분자의 첫번째 항 $q(x_t | x_{t-1}, x_0)$은 $q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1}) = \mathcal{N}(x_{t} ; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t)\mathbf{I})$ 이므로 tractable (Markov property + Eq. 31)  

분자의 두번째 항 $q(x_{t-1}|x_0)$과 분모 $q(x_{t}|x_0)$도 제한조건에 의해 VDM의 encoder transitions가 linear라는 점 때문에 tractable이다.  
이를 보이기 위해 이미 알고 있는 $q(x_t | x_{t-1}) = \mathcal{N}(x_{t} ; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t)\mathbf{I})$에 reparameterization trick을 적용해 $x_t$, $x_{t-1}$를 표현하면,

**Equation 59:**
```math
\begin{aligned}
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon} \quad \text{with } \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\epsilon}; \mathbf{0}, \mathbf{I})
\end{aligned}
```
<br>

> ```math
> x_t\text{ 가 전단계 }x_{t-1}\text{와 노이즈의 단순한 선형결합}
> ```

**Equation 60:**

```math
\begin{aligned}
x_{t-1} = \sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}\boldsymbol{\epsilon} \quad \text{with } \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\epsilon}; \mathbf{0}, \mathbf{I})
\end{aligned}
```
<br>

이 된다.  
2T개의 독립적인 랜덤 노이즈 변수  
```math
\{\bm{\epsilon}^*_t, \bm{\epsilon}_t\}^T_{t=0} \overset{iid}{\sim} \mathcal{N}(\bm{\epsilon}; \bm{0},\textbf{I})
```
가 있고, 그 랜덤 노이즈 변수들에서 샘플링이 가능하다고 가정한 상태에서, 아래와 같이 reparameterization trick으로 표현한 t-1, t-2, ...., 1 단계의 $x_i$들을 계속 대입하면 $x_t \sim q(x_t | x_0)$을 하나의 가우시안 분포로 표현 가능하다.

**Equation 61~70:**

```math
\begin{aligned}
x_t  &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^*\\
&= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}^*\right) + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^*\\
&= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}^* + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^*\\
&= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{\sqrt{\alpha_t - \alpha_t\alpha_{t-1}}^2 + \sqrt{1 - \alpha_t}^2}\boldsymbol{\epsilon}_{t-2} \\
&= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{\alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t}\boldsymbol{\epsilon}_{t-2}\\
&= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2} \\
&= \ldots\\
&= \sqrt{\prod_{i=1}^t\alpha_i}x_0 + \sqrt{1 - \prod_{i=1}^t\alpha_i}\boldsymbol{\epsilon}_0\\
&= \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\boldsymbol{\epsilon}_0 \qquad \bigg(\prod_{i=1}^t\alpha_i \stackrel{\text{def}}{=} \bar\alpha_t \bigg) \\
&\sim \mathcal{N}(x_{t} ; \sqrt{\bar\alpha_t}x_0, \left(1 - \bar\alpha_t\right)\mathbf{I}) 
\end{aligned}
```
<br>

> **(Eq. 63 -> Eq. 66)**
>> ```math
>> \sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}^* + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^*
>> ```
>> 은 서로 독립인 Gaussian의 합이므로 합 결과도 또 Gaussian이 된다.  
>> 이 때, 합 분포의 평균과 분산은 두 Gaussian 분포의 평균, 분산의 합이 된다.   
>> $\sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}^*$의 평균은 $0$, 분산은 $(\sqrt{\alpha_t - \alpha_t\alpha_{t-1}})^2$,  
>> $\sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^*$의 평균은 $0$, 분산은 $(\sqrt{1 - \alpha_t})^2$ 이므로  
>> 합분포의 평균은 $0 + 0 = 0$, 분산은 $(\sqrt{\alpha_t - \alpha_t\alpha_{t-1}})^2 + (\sqrt{1 - \alpha_t})^2 = \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t\alpha_{t-1}$가 된다.  
>> 이를 reparameterization trick을 이용해 노이즈 $\bm{\epsilon}$ 형태로 표현하면 Eq. 66의 $\sqrt{1 - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}$을 얻을 수 있다.

이제 $q(x_{t-1}|x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0)q(x_{t-1}|x_0)}{q(x_{t}|x_0)}$의 우변이 모두 tractable이므로 이들을 대입하면

**Equation 71~84:**

```math
\begin{aligned}
q(x_{t-1}|x_t, x_0)
&= \frac{q(x_t | x_{t-1}, x_0)q(x_{t-1}|x_0)}{q(x_{t}|x_0)}\\
&= \frac{\mathcal{N}(x_{t} ; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t)\mathbf{I})\mathcal{N}(x_{t-1} ; \sqrt{\bar\alpha_{t-1}}x_0, (1 - \bar\alpha_{t-1}) \mathbf{I})}{\mathcal{N}(x_{t} ; \sqrt{\bar\alpha_{t}}x_0, (1 - \bar\alpha_{t})\mathbf{I})}
\end{aligned}
```
```math
\begin{aligned}
&\propto \text{exp}\left\{-\left[\frac{(x_{t} - \sqrt{\alpha_t} x_{t-1})^2}{2(1 - \alpha_t)} + \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0)^2}{2(1 - \bar\alpha_{t-1})} - \frac{(x_{t} - \sqrt{\bar\alpha_t} x_{0})^2}{2(1 - \bar\alpha_t)} \right]\right\}\\
&= \text{exp}\left\{-\frac{1}{2}\left[\frac{(x_{t} - \sqrt{\alpha_t} x_{t-1})^2}{1 - \alpha_t} + \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0)^2}{1 - \bar\alpha_{t-1}} - \frac{(x_{t} - \sqrt{\bar\alpha_t} x_{0})^2}{1 - \bar\alpha_t} \right]\right\}
\end{aligned}
```
```math
\begin{aligned}
&= \text{exp}\left\{-\frac{1}{2}\left[\frac{(-2\sqrt{\alpha_t} x_{t}x_{t-1} + \alpha_t x_{t-1}^2)}{1 - \alpha_t} + \frac{(x_{t-1}^2 - 2\sqrt{\bar\alpha_{t-1}}x_{t-1} x_0)}{1 - \bar\alpha_{t-1}} + C(x_t, x_0)\right]\right\} \\
&\propto \text{exp}\left\{-\frac{1}{2}\left[- \frac{2\sqrt{\alpha_t} x_{t}x_{t-1}}{1 - \alpha_t} + \frac{\alpha_t x_{t-1}^2}{1 - \alpha_t} + \frac{x_{t-1}^2}{1 - \bar\alpha_{t-1}} - \frac{2\sqrt{\bar\alpha_{t-1}}x_{t-1} x_0}{1 - \bar\alpha_{t-1}}\right]\right\}
\end{aligned}
```
```math
\begin{aligned}
&= \text{exp}\left\{-\frac{1}{2}\left[(\frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar\alpha_{t-1}})x_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}x_{t}}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0}{1 - \bar\alpha_{t-1}}\right)x_{t-1}\right]\right\}\\
&= \text{exp}\left\{-\frac{1}{2}\left[\frac{\alpha_t(1-\bar\alpha_{t-1}) + 1 - \alpha_t}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}x_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}x_{t}}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0}{1 - \bar\alpha_{t-1}}\right)x_{t-1}\right]\right\}
\end{aligned}
```
```math
\begin{aligned}
&= \text{exp}\left\{-\frac{1}{2}\left[\frac{\alpha_t-\bar\alpha_{t} + 1 - \alpha_t}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}x_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}x_{t}}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0}{1 - \bar\alpha_{t-1}}\right)x_{t-1}\right]\right\}\\
&= \text{exp}\left\{-\frac{1}{2}\left[\frac{1 -\bar\alpha_{t}}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}x_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}x_{t}}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0}{1 - \bar\alpha_{t-1}}\right)x_{t-1}\right]\right\}
\end{aligned}
```
```math
\begin{aligned}
&= \text{exp}\left\{-\frac{1}{2}\left(\frac{1 -\bar\alpha_{t}}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}\right)\left[x_{t-1}^2 - 2\frac{\left(\frac{\sqrt{\alpha_t}x_{t}}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0}{1 - \bar\alpha_{t-1}}\right)}{\frac{1 -\bar\alpha_{t}}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}}x_{t-1}\right]\right\}\\
&= \text{exp}\left\{-\frac{1}{2}\left(\frac{1 -\bar\alpha_{t}}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}\right)\left[x_{t-1}^2 - 2\frac{\left(\frac{\sqrt{\alpha_t}x_{t}}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0}{1 - \bar\alpha_{t-1}}\right)(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}x_{t-1}\right]\right\}
\end{aligned}
```
```math
\begin{aligned}
&= \text{exp}\left\{-\frac{1}{2}\left(\frac{1}{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}}\right)\left[x_{t-1}^2 - 2\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}x_{t-1}\right]\right\}\\
&\propto \mathcal{N}\left(x_{t-1} ; \underbrace{\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}}_{\boldsymbol{\mu}_q(x_t, x_0)}, \underbrace{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}\mathbf{I}}_{\boldsymbol{\Sigma}_q(t)}\right)
\end{aligned}
```
<br>

이 결과를 보면, $q(x_{t-1}|x_t, x_0)$은 평균이 $\boldsymbol{\mu}_q(x_t, x_0)$이고 분산이 $\boldsymbol{\Sigma}_q(t)$인 Gaussian 분포이므로 tractable함을 알 수 있다.

> 사실 식에 나오는 모든 가우시안 분포들은 다변수 가우시안 분포이므로 지수항은 $-\frac{1}{2}(\mathbf{x}-\bm{\mu})^T\Sigma^{-1}(\mathbf{x}-\bm{\mu})$로 $-\frac{1}{2} \times \text{Mahalanobis distance}(\mathbf{x},\bm{\mu})$이다.  
> 또 모든 분포가 등방성(isotropic, 공분산이 $\mathbf{I}$의 상수배) 이므로 $\Sigma = \sigma^2\mathbf{I} \rightarrow \Sigma^{-1} = \frac{1}{\sigma^2}\mathbf{I}$이 되어  
> 지수항은 $-\frac{1}{2\sigma^2}\|\mathbf{x}-\bm{\mu}\|^2$ 이 된다.  
> 이를 반영해 다시 계산해보면
> ```math
> \begin{aligned}
>q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)
>&= \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_{t}|\mathbf{x}_0)}\\
>&= \frac{\mathcal{N}(\mathbf{x}_{t} ; \sqrt{\alpha_t} \mathbf{x}_{t-1}, (1 - \alpha_t)\mathbf{I})\mathcal{N}(\mathbf{x}_{t-1} ; \sqrt{\bar\alpha_{t-1}}\mathbf{x}_0, (1 - \bar\alpha_{t-1}) \mathbf{I})}{\mathcal{N}(\mathbf{x}_{t} ; \sqrt{\bar\alpha_{t}}\mathbf{x}_0, (1 - \bar\alpha_{t})\mathbf{I})}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&\propto \text{exp}\left\{-\left[\frac{\|\mathbf{x}_{t} - \sqrt{\alpha_t} \mathbf{x}_{t-1}\|^2}{2(1 - \alpha_t)} + \frac{\|\mathbf{x}_{t-1} - \sqrt{\bar\alpha_{t-1}} \mathbf{x}_0\|^2}{2(1 - \bar\alpha_{t-1})} - \frac{\|\mathbf{x}_{t} - \sqrt{\bar\alpha_t} \mathbf{x}_{0}\|^2}{2(1 - \bar\alpha_t)} \right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left[\frac{\|\mathbf{x}_{t} - \sqrt{\alpha_t} \mathbf{x}_{t-1}\|^2}{1 - \alpha_t} + \frac{\|\mathbf{x}_{t-1} - \sqrt{\bar\alpha_{t-1}} \mathbf{x}_0\|^2}{1 - \bar\alpha_{t-1}} - \frac{\|\mathbf{x}_{t} - \sqrt{\bar\alpha_t} \mathbf{x}_{0}\|^2}{1 - \bar\alpha_t} \right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left[\frac{\sum_i(x_{t}^i - \sqrt{\alpha_t} x_{t-1}^i)^2}{1 - \alpha_t} + \frac{\sum_i(x_{t-1}^i - \sqrt{\bar\alpha_{t-1}} x_0^i)^2}{1 - \bar\alpha_{t-1}} - \frac{\sum_i(x_{t}^i - \sqrt{\bar\alpha_t} x_{0}^i)^2}{1 - \bar\alpha_t} \right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left[\frac{\sum_i((x_{t}^i)^2 - 2\sqrt{\alpha_t} x_{t}^i x_{t-1}^i + \alpha_t(x_{t-1}^i)^2)}{1 - \alpha_t} + \frac{\sum_i((x_{t-1}^i)^2 - 2 \sqrt{\bar\alpha_{t-1}} x_{t-1}^i x_0^i + \bar\alpha_{t-1}(x_0^i)^2)}{1 - \bar\alpha_{t-1}} - \frac{\sum_i(x_{t}^i - \sqrt{\bar\alpha_t} x_{0}^i)^2}{1 - \bar\alpha_t} \right]\right\}\\
>& \text{(} \mathbf{x}_{t-1} \text{에 대한 분포이므로 나머지는 상수 취급)} 
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left[\sum_i\bigg(\frac{(-2\sqrt{\alpha_t} x_{t}^ix_{t-1}^i + \alpha_t (x_{t-1}^i)^2)}{1 - \alpha_t} + \frac{((x_{t-1}^i)^2 - 2\sqrt{\bar\alpha_{t-1}}x_{t-1}^i x_0^i)}{1 - \bar\alpha_{t-1}} + C(x_t^i, x_0^i)\bigg)\right]\right\} \\
>&\propto \text{exp}\left\{-\frac{1}{2}\left[\sum_i \bigg(- \frac{2\sqrt{\alpha_t} x_{t}^ix_{t-1}^i}{1 - \alpha_t} + \frac{\alpha_t (x_{t-1}^i)^2}{1 - \alpha_t} + \frac{(x_{t-1}^i)^2}{1 - \bar\alpha_{t-1}} - \frac{2\sqrt{\bar\alpha_{t-1}}x_{t-1}^i x_0^i}{1 - \bar\alpha_{t-1}}\bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left[\sum_i\bigg((\frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar\alpha_{t-1}})(x_{t-1}^i)^2 - 2\left(\frac{\sqrt{\alpha_t}x_{t}^i}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0^i}{1 - \bar\alpha_{t-1}}\right)x_{t-1}^i\bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left[\sum_i\bigg(\frac{\alpha_t(1-\bar\alpha_{t-1}) + 1 - \alpha_t}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}(x_{t-1}^i)^2 - 2\left(\frac{\sqrt{\alpha_t}x_{t}^i}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0^i}{1 - \bar\alpha_{t-1}}\right)x_{t-1}^i\bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left[\sum_i\bigg(\frac{\alpha_t-\bar\alpha_{t} + 1 - \alpha_t}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}(x_{t-1}^i)^2 - 2\left(\frac{\sqrt{\alpha_t}x_{t}^i}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0^i}{1 - \bar\alpha_{t-1}}\right)x_{t-1}^i\bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left[\sum_i\bigg(\frac{1 -\bar\alpha_{t}}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}(x_{t-1}^i)^2 - 2\left(\frac{\sqrt{\alpha_t}x_{t}^i}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0^i}{1 - \bar\alpha_{t-1}}\right)x_{t-1}^i\bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left(\frac{1 -\bar\alpha_{t}}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}\right)\left[\sum_i \bigg( (x_{t-1}^i)^2 - 2\frac{\left(\frac{\sqrt{\alpha_t}x_{t}^i}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0^i}{1 - \bar\alpha_{t-1}}\right)}{\frac{1 -\bar\alpha_{t}}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}}x_{t-1}^i \bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left(\frac{1 -\bar\alpha_{t}}{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}\right)\left[\sum_i \bigg( (x_{t-1}^i)^2 - 2\frac{\left(\frac{\sqrt{\alpha_t}x_{t}^i}{1 - \alpha_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0^i}{1 - \bar\alpha_{t-1}}\right)(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}x_{t-1}^i\bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left(\frac{1}{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}}\right)\left[\sum_i\bigg((x_{t-1}^i)^2 - 2\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t}^i + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0^i}{1 -\bar\alpha_{t}}x_{t-1}^i\bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>& \bigg(\big(\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t}^i + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0^i}{1 -\bar\alpha_{t}}\big) = K \text{로 두면}\bigg) \\
>&= \text{exp}\left\{-\frac{1}{2}\left(\frac{1}{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}}\right)\left[\sum_i\bigg((x_{t-1}^i)^2 - 2Kx_{t-1}^i + K^2 - K^2 \bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&\propto \text{exp}\left\{-\frac{1}{2}\left(\frac{1}{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}}\right)\left[\sum_i\bigg((x_{t-1}^i - K)^2 \bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left(\frac{1}{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}}\right)\left[\sum_i\bigg(\Big(x_{t-1}^i - \big(\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t}^i + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0^i}{1 -\bar\alpha_{t}}\big)\Big)^2 \bigg)\right]\right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&= \text{exp}\left\{-\frac{1}{2}\left(\frac{1}{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}}\right)\|\mathbf{x}_{t-1} - \big(\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\mathbf{x}_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\mathbf{x}_0}{1 -\bar\alpha_{t}}\big)\|^2 \right\}
> \end{aligned}
> ```
> ```math
> \begin{aligned}
>&\propto \mathcal{N}\left(\mathbf{x}_{t-1} ; \underbrace{\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\mathbf{x}_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\mathbf{x}_0}{1 -\bar\alpha_{t}}}_{\boldsymbol{\mu}_q(\mathbf{x}_t, \mathbf{x}_0)}, \underbrace{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}\mathbf{I}}_{\boldsymbol{\Sigma}_q(t)}\right)
> \end{aligned}
> ```
> <br>
> 으로 동일한 결과를 얻을 수 있다.
>
>> 노이즈가 등방성인 가우시안 분포이라는 것은 노이즈가 추가될 때, n차원 벡터 $\mathbf{x}_t$의 각 차원 element(=pixel)에 추가되는 픽셀 노이즈들은 서로 독립으로 correlation이 없어서 공분산 행렬의 대각 성분이 전부 0이라는 것이다.  
>> 따라서 특정 채널의 특정 픽셀에 노이즈를 더 심하게 주지 않고 완벽하게 독립적이고 동일한 강도인 노이즈를 주입한다.

> 추가로 분산 $\boldsymbol{\Sigma}_q(t) = \frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}\mathbf{I}$에서  
>   
> **Equation 85:**
> ```math
> \sigma_q^2(t) = \frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}} 
> ```
> <br>
> 로 두어  
>
> ```math
> \boldsymbol{\Sigma}_q(t) = \sigma_q^2(t)\mathbf{I}
> ```
> <br>
> 로 간략하게 표현한다.

### **$p_{\boldsymbol{\theta}}(x_{t-1}|x_t)$ 모델링 및 학습**

denoising matching term을 통해 $p_{\boldsymbol{\theta}}(x_{t-1}|x_t)$를 $q(x_{t-1}|x_t, x_0)$에 근사하는 것이 학습 목표인 상황.  
$q(x_{t-1}|x_t, x_0)$이 항상 계산 가능한 Gaussian이므로 $p_{\boldsymbol{\theta}}(x_{t-1}|x_t)$도 Gaussian으로 모델링하는게 좋다.  
또 $q(x_{t-1}|x_t, x_0)$의 분산은 t만을 변수로 가지는 함수이므로 각 t step마다 바로 계산 가능하면서 고정된 값이다. 따라서 $p_{\boldsymbol{\theta}}(x_{t-1}|x_t)$의 분산도 $\boldsymbol{\Sigma}_q(t) = \sigma_q^2(t)\mathbf{I}$로 동일하게 설정한다.  
분산은 ground-truth와 동일하게 두지만, 평균 $\bm{\mu}_{\theta}(x_t, t)$는 $p_{\boldsymbol{\theta}}(x_{t-1}|x_t)$이 ground-truth 함수와 다르게 $x_0$를 조건으로 가지지 않으므로 $x_t$에 대한 함수로 설정한다.

두 다변수 가우시안 분포 사이의 KL divergence는

**Equation 86:**

```math
\begin{aligned}
D_{\text{KL}}(\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_x,\boldsymbol{\Sigma}_x) \| \mathcal{N}(\mathbf{y}; \boldsymbol{\mu}_y,\boldsymbol{\Sigma}_y))
&=\frac{1}{2}\left[\log\frac{|\boldsymbol{\Sigma}_y|}{|\boldsymbol{\Sigma}_x|} - d + \text{tr}(\boldsymbol{\Sigma}_y^{-1}\boldsymbol{\Sigma}_x)
+ (\boldsymbol{\mu}_y-\boldsymbol{\mu}_x)^T \boldsymbol{\Sigma}_y^{-1} (\boldsymbol{\mu}_y-\boldsymbol{\mu}_x)\right]
\end{aligned}
```


[wiki 페이지: KL Divergence between two multivariate normal distributions](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions)

이므로 $p_{\boldsymbol{\theta}}(x_{t-1}|x_t)$과 $q(x_{t-1}|x_t, x_0)$ 사이의 KL divergence를 구하면

**Equation 87~92:**

```math
\begin{aligned}
& \quad \arg\min_{\boldsymbol{\theta}} D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t)) \\
&= \arg\min_{\boldsymbol{\theta}} D_{\text{KL}}(\mathcal{N}(x_{t-1}; \boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q(t)) \| \mathcal{N}(x_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}},\boldsymbol{\Sigma}_q(t)))\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2}\left[\log\frac{|\boldsymbol{\Sigma}_q(t)|}{|\boldsymbol{\Sigma}_q(t)|} - d + \text{tr}(\boldsymbol{\Sigma}_q(t)^{-1}\boldsymbol{\Sigma}_q(t))
+ (\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q)^T \boldsymbol{\Sigma}_q(t)^{-1} (\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q)\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2}\left[\log 1 - d + d + (\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q)^T \boldsymbol{\Sigma}_q(t)^{-1} (\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q)\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2}\left[(\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q)^T \boldsymbol{\Sigma}_q(t)^{-1} (\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q)\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2}\left[(\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q)^T \left(\sigma_q^2(t)\mathbf{I}\right)^{-1} (\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q)\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q\right\rVert_2^2\right]
\end{aligned}
```
<br>

> $\bm{\mu}_{q} = \bm{\mu}_{q}(x_t, x_0)$  
> $\bm{\mu}_{\theta} = \bm{\mu}_{\theta}(x_t, t)$

이 된다. 

$\bm{\mu}_{q}$는 아래와 같다. (Eq. 84)

**Equation 93:**

```math
\begin{aligned}
\boldsymbol{\mu}_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}
\end{aligned}
```
<br>

$p_{\boldsymbol{\theta}}(x_{t-1}|x_t)$과 $q(x_{t-1}|x_t, x_0)$ 사이의 KL divergence를 최소화 하는 것은 Eq. 92에서 보이듯 $\bm{\mu}_{q}$와 $\bm{\mu}_{\theta}$사이의 L2 norm을 최소화 하는 것과 동일하므로 $\bm{\mu}_{\theta}$를 $\bm{\mu}_{q}$와 최대한 동일한 형태를 가지도록 모델링 하면,

**Equation 94:**

```math
\begin{aligned}
\boldsymbol{\mu}_{\boldsymbol{\theta}}(x_t, t) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\hat{x}_{\boldsymbol{\theta}}(x_t, t)}{1 -\bar\alpha_{t}}
\end{aligned}
```
<br>

이 된다.

이렇게 모델링한 $\bm{\mu}_{q}$와 $\bm{\mu}_{\theta}$를 Eq. 92의 L2 norm에 대입하면

**Equation 95~99:**

```math
\begin{aligned}
& \quad \arg\min_{\boldsymbol{\theta}} D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t)) \\
&= \arg\min_{\boldsymbol{\theta}} D_{\text{KL}}\left(\mathcal{N}\left(x_{t-1}; \boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q\left(t\right)\right) \| \mathcal{N}\left(x_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}},\boldsymbol{\Sigma}_q\left(t\right)\right)\right)\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\hat{x}_{\boldsymbol{\theta}}(x_t, t)}{1 -\bar\alpha_{t}}-\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}\right\rVert_2^2\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert\frac{\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\hat{x}_{\boldsymbol{\theta}}(x_t, t)}{1 -\bar\alpha_{t}}-\frac{\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}\right\rVert_2^2\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert\frac{\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)}{1 -\bar\alpha_{t}}\left(\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right)\right\rVert_2^2\right]\\
&=\arg\min_{\boldsymbol{\theta}}\frac{1}{2\sigma_q^2(t)}\frac{\bar\alpha_{t-1}(1-\alpha_t)^2}{(1 -\bar\alpha_{t})^2}\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right]
\end{aligned}
```
<br>

노이즈가 끼어 있는 이미지 $x_t$와 time-step $t$값을 가지고 $\hat{x}_{\boldsymbol{\theta}}(x_t, t)$를 원본 $x_0$에 최대한 유사하도록 예측 하는 것으로 바뀌게 된다.

다시 denoising matching term $\sum_{t=2}^T\mathbb{E}_{q(x_{t}|x_0)}\left[D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t))\right] = \sum_{t=2}^T\mathbb{E}_{q(x_{t}|x_0)}\left[\frac{1}{2\sigma_q^2(t)}\frac{\bar\alpha_{t-1}(1-\alpha_t)^2}{(1 -\bar\alpha_{t})^2}\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right]\right]$을 보면  
일반적으로 T=1000으로 둘 때, 이미지 $x_0$한장에 대해 t=2 부터 T까지 999번 모델 신경망이 예측한 $\hat{x}_{\boldsymbol{\theta}}(x_t, t)$과의 L2 norm을 구해서 summation을 진행해야 한다.  
따라서 summation을 그대로 두면 사실상 학습이 불가능할 정도로 엄청난 양의 연산량이 필요해진다.

이를 해결하기 위해서 아래와 같이 식을 변경하면

```math
\begin{aligned}
\sum_{t=2}^T\mathbb{E}_{q(x_{t}|x_0)}\left[D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t))\right]
&= (T-1)\sum_{t=2}^T\frac{1}{T-1}\mathbb{E}_{q(x_{t}|x_0)}\left[D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t))\right] \\
& \bigg(\frac{1}{T-1}\text{은 }t\sim U\{2, T\}\text{의 probability mass }p(t)\text{값}\bigg) \\
&= (T-1)\mathbb{E}_{t\sim U\{2, T\}}\left[\mathbb{E}_{q(x_{t}|x_0)}\left[D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t))\right]\right] \\
&\propto \mathbb{E}_{t\sim U\{2, T\}}\left[\mathbb{E}_{q(x_{t}|x_0)}\left[D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t))\right]\right]
\end{aligned}
```
<br>

로 summation을 $\mathbb{E}_{t\sim U\{2, T\}}$로 변형가능하다.

**Equation 100:**

```math
\begin{aligned}
\arg\min_{\boldsymbol{\theta}}\mathbb{E}_{t\sim U\{2, T\}}\left[\mathbb{E}_{q(x_{t}|x_0)}\left[D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_{\boldsymbol{\theta}}(x_{t-1}|x_t))\right]\right]
\end{aligned}
```
<br>

이렇게 기댓값의 형태로 바꾸고 나면 몬테카를로 샘플링을 사용할 수 있게 된다(L=1).  
그래서 학습 시에는 2부터 T사이에서 무작위로 하나의 time-step t를 뽑고 그 t에 대한 $\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2$만 계산해서 모델 가중치를 업데이트한다.  
이 과정을 수만 장의 이미지 데이터에 대해 수만 번 반복하면, 통계적으로 대수의 법칙에 의해 $\sum_{t=2}^T$ 전체를 계산해서 가중치를 업데이트 한 것과 동일한 결과를 얻을 수 있다.

> (Eq. 99)로 인해 T개의 $p_{\boldsymbol{\theta}}(x_{0}|x_1), p_{\boldsymbol{\theta}}(x_{1}|x_2), \dots , p_{\boldsymbol{\theta}}(x_{t-1}|x_t), \dots , p_{\boldsymbol{\theta}}(x_{T-1}|x_T)$ 분포를 배울 필요 없이 원본 이미지 $x_0$를 예측하는 신경망 모델 $\hat{x}_{\boldsymbol{\theta}}(x_t, t)$ 하나만 학습한다.  
학습 시에 무작위로 time-step t가 정해지면 (Eq. 69 $x_t= \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\boldsymbol{\epsilon}_0$)로 원본 이미지에서 t step 이미지 $x_t$를 바로 생성 가능하다.  
모델은 이 t 단계 이미지와 time-step t를 입력받아 원본 이미지를 복원하고, 이 복원한 이미지와 원본 이미지 사이의 L2 norm 값을 loss로 두고 역전파를 통한 가중치 갱신을 진행한다.
>> (t는 모델에 입력될 때, timestep embedding을 거쳐서 t 단계별 차이를 키워 모델의 학습을 돕는다)

---
---

