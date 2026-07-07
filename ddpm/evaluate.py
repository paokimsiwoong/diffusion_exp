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
            # 여기서 x_0의 근사식은 q(x_t | x_0)로 구해진 x_t = \sqrt{\bar\alpha_t} x_0 +  \sqrt{1 - \bar\alpha_t}\epsilon_0를 
            # x_0에 대해서 정리한뒤 ground truth noise \epsilon_0를 UNet이 예측한 노이즈 \epsilon_\theta로 변경한 것

        """

        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        xt = torch.randn([1, self.image_channels, self.image_size, self.image_size], device=self.device)
        # T단계는 완전한 가우시안(normal) 노이즈

        # Interval to log $\hat{x}_0$
        interval = self.n_steps // n_frames
        # Frames for video
        frames = []

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

            if t_ % interval == 0:
                # Get $\hat{x}_0$ and add to frames
                x0 = self.p_x0(xt, t, eps_theta)

                frames.append(x0[0])
                if not create_video:
                    self.show_image(x0[0], f"{t_}")

            # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
            xt = self.p_sample(xt, t, eps_theta)
            # x_t에서 노이즈를 조금 복원한 x_{t-1}이 다음 for 루프 x_t가 된다

        # Make video
        if create_video:
            self.make_video(frames, self.eval_path / "video.mp4")

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
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """
        # utils의 gather함수 사용 $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    # x_t로부터 x_0의 근사값 \hat{x}_0를 얻는 메소드
    def p_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """
        #### Estimate $x_0$

        $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
         \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        """
        # utils의 gather 함수 사용 $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)

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