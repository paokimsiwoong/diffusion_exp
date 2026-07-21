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
> 이는 diffusion 모델이 마치 손실 압축 처럼 눈에 보이는 중요한 특징들은 잘 남기지만, 눈에 보이지 않는 미세한 디테일들은 무시해서 눈으로 보는 샘플 퀄리티는 좋지만, 픽셀 단위로 미세한 디테일까지 (각 픽셀의 값 1~2의 차이) 얼마나 잘 복원했는가 평가하는 log likelihood 값은 나쁘게 나온다는 것
>> 무손실 압축(lossless compression)을 진행한다고 할 때, 특정 데이터가 압축 된 후의 길이(몇 비트인지)를 codelength라 한다. 이 때, 자주 반복해서 나오는 데이터는 길이가 짧은(작은 codelength) code로 압축하고 거의 나오지 않는 데이터는 긴 code로 변환해야 압축 결과물의 총 code 길이가 짧아진다(용량이 작다)
>>     
>> 섀넌의 정보이론에 따르면 어떤 데이터 $x$가 등장할 확률이 $p(x)$이면, 이 데이터를 무손실 압축하기 위해 필요한 codelength의 이론적 최소값은 
>> $$\text{Codelength} = -\log p(x)$$
>> 이다.   
>> 이 $-\log p(x)$값은 모델을 학습할 때 계산하는 Negative Log-likelihood(NLL)값
>> $$\text{NLL} = -\log p(x)$$
>> 과 완전히 동일하다. 따라서 log likelihood 값으로 모델을 평가한다는 것은 모델이 무손실 압축을 얼마나 잘하고 있는가를 기준으로 평가하는 것이라고 볼 수 있다.  
>> lossless codelengths 대부분이 사람 눈에 보이지 않는 미세한 디테일을 묘사하는데 쓰인다고 하는 것은:  
>>> 이미지 샘플 퀄리티에 중요한 사람 눈에 보이는 중요한 특징들은 데이터셋 내에서 공통된 특징들이므로 등장 확률이 높아($p(x)$값이 크다) 이론적인 codelength 최소값이 작은데 diffusion 모델은 이 특징들을 잘 학습해서(모델 $q(x)$가 $p(x)$값을 잘 예측해서) 짧은 codelength로 압축한다.