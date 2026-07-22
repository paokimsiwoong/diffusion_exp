# [**Denoising Diffusion Probabilistic Models**](../../README.md#table-of-contents)
---
---
---

# [*Introduction*](../../README.md#table-of-contents)
---
---

![celebahq256_4](../imgs/celebahq256_header_image_4x4.png)
![cifar10_400](../imgs/cifar10_eps-fixedlarge-mse_20x20.png)

[이미지 출처](https://arxiv.org/abs/2006.11239)

이 논문은 수학적으로 잘 정의되어 학습이 쉽지만 GAN과 같은 모델들의 이미지 생성 퀄리티에 미치지는 못하던 diffusion 계열 모델을 개선하여 diffusion 모델도 비슷한 수준의 생성 퀄리티를 달성할 수 있다는 것을 보였다. 또 diffusion 모델이 score-based 생성 모델과 연관이 있음을 보인다.   
  
위 예시 사진에서 보이듯 diffusion 모델은 높은 수준의 이미지 생성 퀄리티를 달성했지만, log likelihood 값을 계산해보면 기존의 likelihood-based 모델과 비교해 좋지 않다고 한다. 이 논문은 diffusion 모델의 상대적으로 긴 lossless codelengths 대부분이 사람 눈에 보이지 않는 미세한 디테일을 묘사하는데 쓰이기 때문이라고 주장한다. 
> 이는 diffusion 모델이 마치 손실 압축 처럼 눈에 보이는 중요한 특징들은 잘 압축해서 눈으로 보는 샘플 퀄리티는 좋지만, 눈에 보이지 않는 미세한 디테일들은 무시해서 픽셀 단위로 미세한 디테일까지 (각 픽셀의 값 1~2의 차이) 얼마나 잘 복원했는가 평가하는 log likelihood 값은 나쁘게 나온다는 것
>> 무손실 압축(lossless compression)을 진행한다고 할 때, 특정 데이터가 압축 된 후의 길이(몇 비트인지)를 codelength라 한다. 이 때, 자주 반복해서 나오는 데이터는 길이가 짧은(작은 codelength) code로 압축하고 거의 나오지 않는 데이터는 긴 code로 변환해야 압축 결과물의 총 code 길이가 짧아진다(용량이 작다)
>>     
>> 섀넌의 정보이론에 따르면 어떤 데이터 $x$가 등장할 확률이 $`p(x)`$이면, 이 데이터를 무손실 압축하기 위해 필요한 codelength의 이론적 최소값은 
>> ```math
>> \text{Codelength} = -\log p(x)
>> ```
>> 이다.   
>> 이 $`-\log p(x)`$값은 모델을 학습할 때 계산하는 Negative Log-likelihood(NLL)값
>> ```math
>> \text{NLL} = -\log p(x)
>> ```
>> 과 완전히 동일하다. 따라서 log likelihood 값으로 모델을 평가한다는 것은 모델이 무손실 압축을 얼마나 잘하고 있는가를 기준으로 평가하는 것이라고 볼 수 있다.  
>> lossless codelengths 대부분이 사람 눈에 보이지 않는 미세한 디테일을 묘사하는데 쓰인다고 하는 것은:  
>>> 이미지 샘플 퀄리티에 중요한 사람 눈에 보이는 중요한 특징들은 diffusion 모델이 이 특징들을 잘 학습해서(모델 $`q(x)`$가 $`p(x)`$값을 잘 예측해서) 짧은 codelength로 압축한다. 그러나 픽셀 단위의 국소적인 패턴(질감 - local texture)을 잘 학습하지 못하기 때문에 미세 고주파 노이즈 패턴 $`x`$에 대해 그 패턴의 실제 등장확률 $`p(x)`$ 보다 훨씬 작은 $`q(x)`$값을 할당하게 되고 codelength 결과값이 매우 커지게 된다. 논문의 4.3 절에서 CIFAR 10 데이터셋으로 실험한 결과, 이미지를 압축한 전체 codelengths의 절반 이상이 눈에 보이지 않는 미세 고주파 패턴을 압축한 codelength 부분임을 보인다.

따라서 논문은 diffusion 모델을 무손실 압축을 가정하는 log likelihood 평가 기준 대신 손실 압축의 관점에서 평가해야 한다고 말한다. 이 관점에서 보면 diffusion 모델은 기존의 Autoregressive 모델에서 일반적으로 가능했던 수준을 넘어 광범위하게 일반화된 비트 순서(Bit ordering)를 따르는 자기회귀 디코딩과 유사한 점진적 디코딩이라는 것.
> bit ordering은 어떤 정보를 우선적으로 압축하는지를 정하는 순서 기준
>> 기존의 PixelCNN 같은 모델들은 spatial ordering으로 첫번째 픽셀 -> 그 옆 두번째 픽셀 -> 그 옆 세번째 픽셀 -> ... 와 같이 공간적인 위치를 기준으로 순서를 정해 압축을 진행한다. 이러한 방식은 픽셀 단위의 미세한 질감은 잘 맞추지만 전체적인 형태를 잘 맞추지 못해 눈이 3개가 된다던가, 형태가 일그러진다던가 하는 문제들이 발생한다.
>  
>> diffusion 모델은 처음부터 이미지 전체 크기의 노이즈에서 시작해 T step 동안 점진적으로 이미지 전체를 변화시키는 방식으로 초반 t step (t ~ T)에는 대상의 윤곽, 배경의 대략적인 색상 등 중요한 뼈대가 되는 거시적 정보부터 먼저 보고 후반 t step (t ~ 0) 에 마지막으로 피부 질감, 매끈한 표면의 반사광 등 미세한 디테일을 처리한다. 이러한 bit ordering은 인간이 그림을 그리는 방식에 더 유사한 정보의 순서 기준이라고 볼 수 있다.  
> 
> 따라서 논문에서 말하는 광범위하게 일반화된 비트 순서(Bit ordering)이란 단순히 픽셀 순서대로 정하는 것이 아닌 정보의 중요도를 기준으로 하는 고차원적인 비트 순서를 말한다.

---
## [*Codelength, entropy, cross entropy*](../../README.md#table-of-contents)

