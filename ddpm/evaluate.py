"""
출처: https://nn.labml.ai/diffusion/ddpm/evaluate.html
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize

from tqdm import tqdm
from pathlib import Path
import yaml
from datetime import datetime

from __init__ import DenoiseDiffusion
from utils import gather
from experiment import Configs, DDPM


class Sampler:
    """
    ## Sampler class
    """

    def __init__(self, ddpm: DDPM):
        """
        * `diffusion` is the `DenoiseDiffusion` instance
        * `image_channels` is the number of channels in the image
        * `image_size` is the image size
        * `device` is the device of the model
        # @@@ DDPM 클래스 입력으로 변경해 위 4개 변수 다 DDPM 인스턴스에서 정보 가져오기
        """
        self.ddpm = ddpm

        self.device = ddpm.cfg.device
        self.image_size = ddpm.cfg.image_size
        self.image_channels = ddpm.cfg.image_channels
        self.diffusion = ddpm.diffusion

        self.exp_path = ddpm.exp_path
        self.eval_path = ddpm.exp_path / "eval"
        self.eval_path.mkdir(parents=True, exist_ok=True)

        # $T$
        self.n_steps = self.diffusion.n_steps
        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        self.eps_model = self.diffusion.eps_model

        # $\beta_t$
        self.beta = self.diffusion.beta
        # torch.linspace(0.0001, 0.02, n_steps).to(device)

        # $\alpha_t$
        self.alpha = self.diffusion.alpha
        # 1 - beta

        # $\bar\alpha_t$
        self.alpha_bar = self.diffusion.alpha_bar
        # torch.cumprod(alpha, dim=0)
        # (α_0, α_1, α_2, ..., α_i, ...) ==> (α_0, α_0*α_1, α_0*α_1*α_2, ..., α_0*α_1*....*α_i, ...)

        # beta, alpha, alpha_bar 모두 torch.Size([n_steps])

        # $\bar\alpha_{t-1}$
        alpha_bar_tm1 = torch.cat([self.alpha_bar.new_ones((1,)), self.alpha_bar[:-1]])
        # alpha_bar는 [\bar\alpha_0, \bar\alpha_1, \bar\alpha_2, ...., \bar\alpha_t, ...., \bar\alpha_n_steps]로 되어 있다.
        # alpha_bar_tm1는 [1, \bar\alpha_0, \bar\alpha_1, ...., \bar\alpha_(t-1), ...., \bar\alpha_(n_steps-1)]로
        # 같은 step에 alpha_bar가 \bar\alpha_t일 때, alpha_bar_tm1은 \bar\alpha_(t-1)이다.
            # tensor.new_ones(size)는 tensor와 동일한 dtype, device로 설정된다
            # 지정한 size에 1 값을 채워서 반환
            # dtype과 device를 다르게 하고 싶을 때는 따로 지정 가능


        # To calculate
        #
        # \begin{align}
        # q(x_{t-1}|x_t, x_0) &= \mathcal{N} \Big(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t \mathbf{I} \Big) \\
        # \tilde\mu_t(x_t, x_0) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
        #                          + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\
        # \tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t
        # \end{align}
            # q(x_{t-1}|x_t, x_0) 계산식

        # $$\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$$
        self.beta_tilde = self.beta * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        # self.beta_tilde는 q(x_{t-1}|x_t, x_0)의 분산

        # q(x_{t-1}|x_t, x_0)의 평균은 x_0와 x_t의 선형결합

        # $$\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$$
        self.mu_tilde_coef1 = self.beta * (alpha_bar_tm1 ** 0.5) / (1 - self.alpha_bar)
        # x_0의 계수

        # $$\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1}}{1-\bar\alpha_t}$$
        self.mu_tilde_coef2 = (self.alpha ** 0.5) * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        # x_t의 계수

        # $\sigma^2 = \beta$
        self.sigma2 = self.beta
        # q(x_t | x_{t-1})의 분산 
            # q(x_t | x_{t-1})는 평균이 √α_t * x_{t-1}이고 분산이 (1-α_t)I
            # 1-α_t = β_t

    def show_image(self, img, title=""):
        """Helper function to display an image"""
        img = img.clip(0, 1)
        # tensor.clip == tensor.clamp
            # clamp(min, max)는 min보다 작은 값은 min로, max보다 큰 값은 max로 바꿔서
            # [min, max] 범위로 텐서 값을 제한한다.  
        img = img.cpu().numpy()

        plt.imshow(img.transpose(1, 2, 0))
        # [c, h, w] 를 [h, w, c]로 transpose 한 후 plt에 입력
        plt.title(title)
        plt.show()

    def make_video(self, frames, path: Path = Path("video.mp4")):
        """Helper function to create a video"""
        import imageio
        # 20 second video
        writer = imageio.get_writer(path, fps=len(frames) // 20)
        # Add each image
        for f in frames:
            f = f.clip(0, 1)
            # tensor.clip == tensor.clamp
                # clamp(min, max)는 min보다 작은 값은 min로, max보다 큰 값은 max로 바꿔서
                # [min, max] 범위로 텐서 값을 제한한다.  

            f = to_pil_image(resize(f, [368, 368]))
            writer.append_data(np.array(f))
        #
        writer.close()

    def sample_animation(self, n_frames: int = 1000, create_video: bool = True):
        """
        #### Sample an image step-by-step using $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        We sample an image step-by-step using $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$ and at each step
        show the estimate
        $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
         \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
            # 여기서 x_0의 근사식(\hat{x}_0)은 q(x_t | x_0)로 구해진 x_t = \sqrt{\bar\alpha_t} x_0 +  \sqrt{1 - \bar\alpha_t}\epsilon_0를 
            # x_0에 대해서 정리한뒤 ground truth noise \epsilon_0를 UNet이 예측한 노이즈 \epsilon_\theta로 변경한 것

        """

        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        xt = torch.randn([1, self.image_channels, self.image_size, self.image_size], device=self.device)
        # T단계는 완전한 가우시안(normal) 노이즈

        # Interval to log $\hat{x}_0$
        interval = self.n_steps // n_frames
        # Frames for video
        frames = []

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # x_t도 어떻게 변화해가는지 확인해보기
        frames_xt = []
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # Sample $T$ steps
        for t_inv in tqdm(
            range(self.n_steps),
            total=self.n_steps
        ):
            # $t$
            t_ = self.n_steps - t_inv - 1
            # T-1 ~ 0

            # $t$ in a tensor
            t = xt.new_full((1,), t_, dtype=torch.long)
            # tensor.new_full(size, fill_value)는 tensor와 동일한 dtype, device
            # 지정한 size와 fill_value로 새 텐서를 생성
            # dtype과 device를 다르게 하고 싶을 때는 따로 지정 가능

            # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
            eps_theta = self.eps_model(xt, t)
            # t-step의 노이즈 예측
            # xt는 [b=1, c=3, h, w]
            # t는 [b=1]

            if t_ % interval == 0:
                # Get $\hat{x}_0$ and add to frames
                x0 = self.p_x0(xt, t, eps_theta)

                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                # 여기서 x_t 대신 t step에서 구한 x_0의 근사값 \hat{x}_0를 시각화 하는 이유
                    # x_t는 t 값이 클 때 (완전 가우시안 노이즈인 T와 가까울 때) 거의 가우시안 노이즈와 유사해
                    # x_t로 복원 과정을 시각화하면 상당 부분이 거의 노이즈만 낀 프레임들을 보게 되어 시각화의 의미가 거의 없다.
                        # 실제로 생성된 20초 영상을 확인해보면 
                        # 거의 15초가 되어서 흐릿하게 얼굴 윤곽선이 보이기 전까진 무의미해보이는 노이즈만 낀 프레임만 보이고
                        # 마지막 몇초 사이에 급격하게 복원이 진행되는 것처럼 보인다.
                    # 그대신 각 t step마다 x_0의 근사값을 계산해서 보면
                        # \hat{x}_0는 현재 step에서 가진 정보로 예측하는 원본 이미지이므로
                        # 각 단계에서 예측하는 이미지가 얼마나 원본 이미지에 가까운지를 확인하면
                        # 현재 복원이 잘 진행되고 있는지 시각적으로 확인 가능하다.
                            # 실제로 생성된 x_0 시각화 영상을 보면
                            # 2~3초 시점부터 사람 얼굴로 인식 가능한 프레임이 보이기 시작한다.
                            # @@@ 거의 최후반 전까지는 예측 결과가 최종 결과와는 다른 사람들로 계속 바뀌다가 
                            # @@@ 마지막 1~2초에 한사람으로 결정되는 것처럼 보인다.
                    # 초기 단계에는 큰 부분 부분별 색상이나 큰 윤곽선이 보이는 저주파 특징이 보이고
                    # 후반에는 머리카락, 텍스쳐, 세세한 경계선 등의 디테일이 보이는 고주파 특징을 확인 가능하다.
                frames_xt.append(xt[0]) # @@@ x_t도 리스트에 저장해 x_t의 변화과정 확인
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                frames.append(x0[0])
                # x0[0]을 하면 [b=1, c=3, h, w]가 [c=3, h, w]로 바뀌어 squeeze(0)로 배치차원을 제거하는 효과

                if not create_video:
                    self.show_image(x0[0], f"{t_}")

            # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
            xt = self.p_sample(xt, t, eps_theta)
            # x_t에서 노이즈를 조금 복원한 x_{t-1}이 다음 for 루프 x_t가 된다

        # Make video
        if create_video:
            self.make_video(frames, self.eval_path / "video.mp4")
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            self.make_video(frames_xt, self.eval_path / "video_xt.mp4")
            # x_t 비디오 파일 생성
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, lambda_: float, t_: int = 100):
        """
        #### Interpolate two images $x_0$ and $x'_0$

        We get $x_t \sim q(x_t|x_0)$ and $x'_t \sim q(x'_t|x_0)$.

        Then interpolate to
         $$\bar{x}_t = (1 - \lambda)x_t + \lambda x'_0$$

        Then get
         $$\bar{x}_0 \sim \textcolor{lightgreen}{p_\theta}(x_0|\bar{x}_t)$$

        * `x1` is $x_0$
        * `x2` is $x'_0$
        * `lambda_` is $\lambda$
        * `t_` is $t$
        """

        # Number of samples
        n_samples = x1.shape[0]
        # $t$ tensor
        t = torch.full((n_samples,), t_, device=self.device)
        # $$\bar{x}_t = (1 - \lambda)x_t + \lambda x'_0$$
        xt = (1 - lambda_) * self.diffusion.q_sample(x1, t) + lambda_ * self.diffusion.q_sample(x2, t)

        # $$\bar{x}_0 \sim \textcolor{lightgreen}{p_\theta}(x_0|\bar{x}_t)$$
        return self._sample_x0(xt, t_)

    def interpolate_animate(self, x1: torch.Tensor, x2: torch.Tensor, n_frames: int = 100, t_: int = 100,
                            create_video=True):
        """
        #### Interpolate two images $x_0$ and $x'_0$ and make a video

        * `x1` is $x_0$
        * `x2` is $x'_0$
        * `n_frames` is the number of frames for the image
        * `t_` is $t$
        * `create_video` specifies whether to make a video or to show each frame
        """

        # Show original images
        self.show_image(x1, "x1")
        self.show_image(x2, "x2")
        # Add batch dimension
        x1 = x1[None, :, :, :]
        x2 = x2[None, :, :, :]
        # $t$ tensor
        t = torch.full((1,), t_, device=self.device)
        # $x_t \sim q(x_t|x_0)$
        x1t = self.diffusion.q_sample(x1, t)
        # $x'_t \sim q(x'_t|x_0)$
        x2t = self.diffusion.q_sample(x2, t)

        frames = []
        # Get frames with different $\lambda$
        # for i in monit.iterate('Interpolate', n_frames + 1, is_children_silent=True):
            # is_children_silent=True이면 이 루프의 진행률만 화면에 표시하고 
            # 루프 안의 nested loop가 있을 경우 그 진행률은 무시 ==> _sample_x0의 진행률 바 미 표시
        for i in tqdm(
            range(n_frames + 1),
            desc= 'Interpolate'
        ):
            # $\lambda$
            lambda_ = i / n_frames
            # $$\bar{x}_t = (1 - \lambda)x_t + \lambda x'_0$$
            xt = (1 - lambda_) * x1t + lambda_ * x2t
            # $$\bar{x}_0 \sim \textcolor{lightgreen}{p_\theta}(x_0|\bar{x}_t)$$
            x0 = self._sample_x0(xt, t_)
            # Add to frames
            frames.append(x0[0])
            # Show frame
            if not create_video:
                self.show_image(x0[0], f"{lambda_ :.2f}")

        # Make video
        if create_video:
            self.make_video(frames, self.eval_path / "video.mp4")

    def _sample_x0(self, xt: torch.Tensor, n_steps: int):
        """
        #### Sample an image using $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        * `xt` is $x_t$
        * `n_steps` is $t$
        """

        # Number of sampels
        n_samples = xt.shape[0]
        # Iterate until $t$ steps
        for t_ in range(n_steps):
        # 이 루프는 interpolate_animate 함수 안에서 도는 loop 안의 루프이므로
        # tqdm 미사용
            t = n_steps - t_ - 1
            # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
            xt = self.diffusion.p_sample(xt, xt.new_full((n_samples,), t, dtype=torch.long))

        # Return $x_0$
        return xt

    def sample(self, n_samples: int = 16):
        """
        #### Generate images
        """
        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        xt = torch.randn([n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)

        # $$x_0 \sim \textcolor{lightgreen}{p_\theta}(x_0|x_t)$$
        x0 = self._sample_x0(xt, self.n_steps)

        # Show images
        for i in range(n_samples):
            self.show_image(x0[i])

    # x_t에서 노이즈를 조금 복원한 x_{t-1}를 생성하는 메소드
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, eps_theta: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
            # DDPM 논문에서 p_θ(x_{t-1}|x_t)의 분산은 ground truth q(x_{t-1} | x_t, x_0)의 복잡한 분산을 그대로 쓰지 않고
            # q(x_t | x_{t-1})의 간단한 분산을 사용해도 결과에 차이가 없다는 것을 보였으므로 
            # q(x_t | x_{t-1})의 분산 \sigma_t^2(self.sigma2)을 사용 
            # (자세한 설명은 __init__.py의 DenoiseDiffusion 클래스 p_sample 메소드 주석 확인)
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """
        # utils의 gather함수 사용 $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # self.alpha_bar는 torch.Size([n_steps])
        # t는 torch.Size([1])
        # 지정한 t step의 alpha_bar값만 gather로 추출       

        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # alpha도 지정한 t step의 값만 추출

        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # p_θ(x_{t-1}|x_t)의 평균 계산

        # $\sigma^2$
        var = gather(self.sigma2, t)
        # sigma2도 지정한 t step의 값만 추출

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # N(0, I)에서 노이즈 샘플링 

        # Sample
        return mean + (var ** .5) * eps
        # 평균 0, 분산 1인 노이즈로 평균 mean, 분산 var인 x_{t-1}를 생성

    # x_t로부터 x_0의 근사값 \hat{x}_0를 얻는 메소드
    def p_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """
        #### Estimate $x_0$

        $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
         \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        
            # 여기서 x_0의 근사식(\hat{x}_0)은 q(x_t | x_0)로 구해진 x_t = \sqrt{\bar\alpha_t} x_0 +  \sqrt{1 - \bar\alpha_t}\epsilon_0를 
            # x_0에 대해서 정리한뒤 ground truth noise \epsilon_0를 UNet이 예측한 노이즈 \epsilon_\theta로 변경한 것
        """
        # utils의 gather 함수 사용 $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # self.alpha_bar는 torch.Size([n_steps])
        # t는 torch.Size([1])
        # 지정한 t step의 alpha_bar값만 gather로 추출

        # $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
        #  \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        return (xt - (1 - alpha_bar) ** 0.5 * eps) / (alpha_bar ** 0.5)


def main():
    """Generate samples"""

    exp_root = input("eval을 진행할 실험 폴더 경로를 입력하세요. : ")
    print(exp_root)

    exp_path = Path(exp_root)

    if not exp_path.exists():
        print("잘못된 경로 입니다.")
        print("실험 종료")
        return
    elif not exp_path.is_dir():
        print("잘못된 경로 입니다.")
        print("실험 종료")
        return

    print("".center(100, "-"))

    yaml_path = exp_path / "exp.yaml"
    pth_path = list(exp_path.glob("*_best.pth"))[0]
    # TODO: best대신 latest를 사용? 또는 둘 중 선택 가능하게 하기

    with open(yaml_path, "r") as f:
        loaded_dict = yaml.safe_load(f)
        # Create configurations
        configs = Configs(**loaded_dict)

    time_start = datetime.now()
    eval_start = time_start.strftime("%Y%m%d_%H%M%S")

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    configs.wandb = 'disabled'
    # TODO: 학습과정에 추가된 run에 evaluate에서 생성한 video 파일만 추가 가능한지 찾아보기
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ddpm = DDPM(configs)

    load_dict = torch.load(pth_path, map_location="cpu")

    ddpm.eps_model.load_state_dict(load_dict["model_state_dict"])

    ddpm.eps_model.eval()

    # Create sampler
    # sampler = Sampler(diffusion=ddpm.diffusion,
    #                   image_channels=configs.image_channels,
    #                   image_size=configs.image_size,
    #                   exp_path=ddpm.exp_path,
    #                   device=torch.device(configs.device))
    sampler = Sampler(ddpm=ddpm)

    init_end = datetime.now()
    init_time = init_end - time_start
    init_time = str(init_time).split(".")[0]
    print(f"==>> initialization_time: {init_time}")

    print("".center(100, "-"))


    # Start evaluation
    with torch.no_grad():
    # No gradients

        # Sample an image with an denoising animation
        sampler.sample_animation()

        if False:
            # Get some images from data
            data = next(iter(configs.data_loader)).to(configs.device)

            # Create an interpolation animation
            sampler.interpolate_animate(data[0], data[1])

    print("".center(100, "-"))
    print("".center(100, "-"))
    print("".center(100, "-"))

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]
    print(f"==>> total time: {total_time}")

    print("".center(100, "-"))
    print("".center(100, "-"))
    print("".center(100, "-"))

#
if __name__ == '__main__':
    main()