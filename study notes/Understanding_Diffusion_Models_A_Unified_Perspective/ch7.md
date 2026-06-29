
# [**Understanding Diffusion Models: A Unified Perspective**](../Table_of_contents.md#table-of-contents)
---
---
---

# [*Learning Diffusion Noise Parameters*](../Table_of_contents.md#table-of-contents)
---
---

> $\alpha_t$를 학습하지 않는 세팅으로 코드 구현 예정이므로 일단 생략

**Equation 101~108:**

```math
\begin{aligned}
\frac{1}{2\sigma_q^2(t)}\frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{(1 -\bar{\alpha}_{t})^2}\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right]
&= \frac{1}{2\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 -\bar{\alpha}_{t}}}\frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{(1 -\bar{\alpha}_{t})^2}\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right]
\end{aligned}
```
```math
\begin{aligned}
&= \frac{1}{2}\frac{1 -\bar{\alpha}_{t}}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}\frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{(1 -\bar{\alpha}_{t})^2}\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right]\\
&= \frac{1}{2}\frac{\bar{\alpha}_{t-1}(1-\alpha_t)}{(1 - \bar{\alpha}_{t-1})(1 -\bar{\alpha}_{t})}\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right]
\end{aligned}
```
```math
\begin{aligned}
&= \frac{1}{2}\frac{\bar{\alpha}_{t-1}-\bar{\alpha}_t}{(1 - \bar{\alpha}_{t-1})(1 -\bar{\alpha}_{t})}\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right]\\
&= \frac{1}{2}\frac{\bar{\alpha}_{t-1} - \bar{\alpha}_{t-1}\bar{\alpha}_t + \bar{\alpha}_{t-1}\bar{\alpha}_t-\bar{\alpha}_t}{(1 - \bar{\alpha}_{t-1})(1 -\bar{\alpha}_{t})}\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right]
\end{aligned}
```
```math
\begin{aligned}
&= \frac{1}{2}\frac{\bar{\alpha}_{t-1}(1 - \bar{\alpha}_t) -\bar{\alpha}_t(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_{t-1})(1 -\bar{\alpha}_{t})}\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right]\\
&= \frac{1}{2}\left(\frac{\bar{\alpha}_{t-1}(1 - \bar{\alpha}_t)}{(1 - \bar{\alpha}_{t-1})(1 -\bar{\alpha}_{t})} -\frac{\bar{\alpha}_t(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_{t-1})(1 -\bar{\alpha}_{t})}\right)\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right] \\
&= \frac{1}{2}\left(\frac{\bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t-1}} -\frac{\bar{\alpha}_t}{1 -\bar{\alpha}_{t}}\right)\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right]
\end{aligned}
```
<br>

**Equation 109:**

```math
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 -\bar{\alpha}_{t}} 
```
<br>

**Equation 110:**

```math
\frac{1}{2\sigma_q^2(t)}\frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{(1 -\bar{\alpha}_{t})^2}\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right] = \frac{1}{2}\left(\text{SNR}(t-1) -\text{SNR}(t)\right)\left[\left\lVert\hat{x}_{\boldsymbol{\theta}}(x_t, t) - x_0\right\rVert_2^2\right] 
```
<br>

**Equation 111:**

```math
\text{SNR}(t) = \exp(-\omega_{\boldsymbol{\eta}}(t)) 
```
<br>

**Equation 112~114:**

```math
\begin{aligned}
&\frac{\bar{\alpha}_t}{1 -\bar{\alpha}_{t}} = \exp(-\omega_{\boldsymbol{\eta}}(t))\\
&\therefore \bar{\alpha}_t = \text{sigmoid}(-\omega_{\boldsymbol{\eta}}(t))\\
&\therefore 1 - \bar{\alpha}_t = \text{sigmoid}(\omega_{\boldsymbol{\eta}}(t))
\end{aligned}
```
<br>

---
---

