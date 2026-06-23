"""
---
title: Denoising Diffusion Probabilistic Models (DDPM) training
summary: >
  Training code for
  Denoising Diffusion Probabilistic Model.
---

# [Denoising Diffusion Probabilistic Models (DDPM)](index.html) training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/experiment.ipynb)

This trains a DDPM based model on CelebA HQ dataset. You can find the download instruction in this
[discussion on fast.ai](https://forums.fast.ai/t/download-celeba-hq-dataset/45873/3).
Save the images inside [`data/celebA` folder](#dataset_path).

The paper had used a exponential moving average of the model with a decay of $0.9999$. We have skipped this for
simplicity.
"""
from typing import List, Literal
from pathlib import Path

import torchvision
from PIL import Image

import numpy as np

import torch
import torch.utils.data
from __init__ import DenoiseDiffusion
from unet import UNet

from dataclasses import dataclass, field, asdict
import yaml

from tqdm import tqdm
import wandb
import os

from datetime import datetime

@dataclass
class Configs:

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset 경로
    root: str = "../datasets"
    # 실험에 사용할 dataset
    dataset: str = "celebA"
    # dataset: str = "MNIST"

    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    image_size: int = 32
    # Number of channels in the initial feature map
    n_channels: int = 64
    # The list of channel numbers at each resolution.

    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = field(default_factory=lambda:[1, 2, 2, 4])
    # @@@ dataclass는 list와 같은 mutable 바로 default 지정 불가능
    # @@@ field(default_factory=List)와 같이 기본값을 생성하는 함수를 지정

    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[bool] = field(default_factory=lambda:[False, False, False, True])

    # Number of time steps $T$
    n_steps: int = 1_000
    # early stopping 기준 횟수
    early_stop: int = 20
    # Batch size
    batch_size: int = 64
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 2e-5

    # Number of training epochs
    epochs: int = 1_000

    pth_folder: str = "./pths"
    sample_folder: str = "./samples"

    wandb: Literal['online', 'offline', 'disabled', 'shared'] = "online"
    wandb_project_name: str = "diffusion"
    wandb_entity: str = ""
    wandb_log_name: str = ""


class DDPM:
    def __init__(self, cfg: Configs):
        self.cfg = cfg
        # Create ε_θ(x_t, t) model
        self.eps_model = UNet(
            image_channels=cfg.image_channels,
            n_channels=cfg.n_channels,
            ch_mults=cfg.channel_multipliers,
            is_attn=cfg.is_attention,
        ).to(cfg.device)

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=cfg.n_steps,
            device=torch.device(cfg.device),
        )

        # dataset
        if cfg.dataset == "celebA":
            self.dataset = CelebADataset(cfg.root, cfg.image_size)
        elif cfg.dataset != "celebA" and cfg.dataset == "MNIST":
            self.dataset = MNISTDataset(cfg.root, cfg.image_size)
        else:
            print("데이터셋 이름 지정 오류")
            raise ValueError


        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, cfg.batch_size, shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=cfg.learning_rate)

        # wandb 초기화
        self.wandb_run = wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config={
            "lr": cfg.learning_rate,
            "dataset": "CelebA or MNIST",
            "n_epochs": cfg.epochs,
            "loss": "MSE",
            "notes": "ddpm 실험",
        },
        name=cfg.wandb_log_name,
        mode=cfg.wandb, 
        )
        wandb.watch((self.eps_model,))

        # 이번 로그 샘플 이미지 저장 폴더 생성
        os.makedirs(f"{cfg.sample_folder}/{cfg.wandb_log_name}", exist_ok=True)

    def sample(self, e:int):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([self.cfg.n_samples, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size],
                            device=self.cfg.device)

            # Remove noise for $T$ steps
            for t_ in tqdm(
                range(self.cfg.n_steps),
                total=self.cfg.n_steps
            ):
                # $t$
                t = self.cfg.n_steps - t_ - 1
                # T 부터 시작해 0까지 복원하므로 self.cfg.n_steps - t_
                # 마지막 단계가 0이 되려면 -1 추가로 빼야함

                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.cfg.n_samples,), t, dtype=torch.long))
                # tensor.new_full(size, fill_value)는 tensor와 동일한 dtype, device
                # 지정한 size와 fill_value로 새 텐서를 생성
                # dtype과 device를 다르게 하고 싶을 때는 따로 지정 가능

            # Log samples
            x_cpu = x.detach().cpu()
            img_array = torchvision.utils.make_grid(x_cpu, nrow=8, normalize=True, value_range=(-1, 1))
            if self.cfg.wandb != "disabled":
                images = wandb.Image(img_array)
                self.wandb_run.log({"samples": images})

            torchvision.utils.save_image(img_array, f"{self.cfg.sample_folder}/{self.cfg.wandb_log_name}/epoch_{e}.png")



    def train(self):
        """
        ### Train
        """

        epoch_loss = 0

        num_batches_train = len(self.data_loader)

        # Iterate through the dataset
        for step, data in tqdm(
            enumerate(self.data_loader),
            total=num_batches_train,
        ):
            # Move data to device
            data = data.to(self.cfg.device)

            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)

            loss_value = loss.item()

            epoch_loss += loss_value

            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()

            # Track the loss
            if self.cfg.wandb != "disabled":
                wandb_step_dict = {
                    "step_loss": loss_value,
                }

                self.wandb_run.log(wandb_step_dict)

        # 배치당 loss 값의 평균 계산
        epoch_mean_batch_loss = epoch_loss / num_batches_train

        return epoch_mean_batch_loss

    def run(self):
        """
        ### Training loop
        """

        # early stopping에 사용할 best loss 값
        best_loss = np.inf
        # TODO: resume일 경우 best_loss 저장된 값으로 변경
        counter = 0
        # TODO: resume일 경우 counter 저장된 값으로 변경

        for e in range(self.cfg.epochs):
            
            print("".center(100, "-"))
            print("".center(100, "-"))
            print(f"Start train #{e+1:2d}")
            epoch_start = datetime.now()

            # Train the model
            epoch_mean_batch_loss = self.train()

                

            train_end = datetime.now()
            train_time = train_end - epoch_start
            train_time = str(train_time).split(".")[0]
            print("".center(100, "-"))
            print("".center(100, "-"))
            print(
                f"==>> epoch {e+1} train_time: {train_time}"
            )
            print("".center(100, "-"))
            print(
                f"mean_batch_loss: {round(epoch_mean_batch_loss,6)}"
            )
            if best_loss > epoch_mean_batch_loss:
                print("".center(100, "-"))
                print(
                    f"Best loss performance at epoch: {e+1}, {round(best_loss, 6)} -> {round(epoch_mean_batch_loss,6)}"
                )
                best_loss = epoch_mean_batch_loss
                counter = 0
            else:
                counter += 1

            # pth 파일 저장 경로
            # ckpt_fpath = osp.join(self.cfg.pth_folder, f"ddpm_{self.cfg.wandb_log_name}_latest.pth")
                # pathlib 사용하는 방식으로 변경
            ckpt_dpath = Path(self.cfg.pth_folder)
            ckpt_fpath = ckpt_dpath / f"ddpm_{self.cfg.wandb_log_name}_latest.pth"

            # configs 클래스 dict 변환
            cfg_dict = asdict(self.cfg)
            cfg_dict["model_state_dict"] = self.eps_model.state_dict()
            cfg_dict["optimizer_state_dict"] = self.optimizer.state_dict()
            cfg_dict["current_epoch"] = e
            cfg_dict["best_loss"] = best_loss
            cfg_dict["early_stop_counter"] = counter
            # TODO: pth에 yaml정보를 다 넣는대신 yaml을 매번 따로 생성하는 방식으로 변경?

            torch.save(cfg_dict, ckpt_fpath)
            # @@@ torch.save는 Path 오브젝트 지원

            # counter가 0이면 best loss
            if counter == 0:
                # best_fpath = osp.join(self.cfg.pth_folder, f"ddpm_{self.cfg.wandb_log_name}_best.pth")
                best_fpath = ckpt_dpath / f"ddpm_{self.cfg.wandb_log_name}_best.pth"
                torch.save(cfg_dict, best_fpath)
                # TODO: validation loss의 best가 아닌데 best 모델 저장할 필요가 있을지?

            print("".center(100, "-"))
            print("".center(100, "-"))
            print(f"Start sampling #{e+1:2d}")
            sp_start = datetime.now()

            # TODO: sample 횟수 조절하기(1에폭당 한번 대신 10~100에폭당 1번?)
            # Sample some images
            self.sample(e)
            
            epoch_end = datetime.now()
            epoch_time = epoch_end - epoch_start
            epoch_time = str(epoch_time).split(".")[0]
            print("".center(100, "-"))
            print("".center(100, "-"))
            print(
                f"==>> epoch {e+1} time: {epoch_time}"
            )

            # loss 수렴 확인될 경우 epoch 강제종료
            if counter == self.cfg.early_stop:
                print("".center(100, "-"))
                print("".center(100, "-"))
                print(f"Weight loss plateau reached at epoch{e-counter}.\nBest loss: {round(best_loss,4)}\nEalry stopping.")
                break


class CelebADataset(torch.utils.data.Dataset):
    """
    ### CelebA HQ dataset

    labml.lab을 안쓰는 방향으로 수정
    """

    def __init__(self, root:str, image_size: int):
        super().__init__()

        # datasets 폴더
        root_dir = Path(root)

        # CelebA images folder
        folder = root_dir / 'celebA'
        # pathlib.Path는 /를 사용해 편하게 경로를 합칠 수 있다.
            # cf: os.path.join(root_dir, 'celebA')

        # List of files
        self._files = [p for p in folder.glob(f'**/*.jpg')]
        # glob은 폴더 내에서 지정한 패턴에 맞는 파일들을 찾아주는 Path 객체의 메소드
            # *.jpg는 현재 폴더만 탐색 (폴더 안에 하위 폴더가 있어도 그 하위 폴더 안은 미확인)
            # */*.jpg는 하위 폴더가 있으면 그 폴더 안의 파일을 탐색 (현재 폴더 안 파일은 무시, 하위 폴더 안에 추가적인 하위 폴더가 있으면 무시)
            # **/*.jpg는 현재 폴더 안 파일과 모든 깊이의 하위 폴더 전부 탐색 

        # Transformations to resize the image and convert to tensor
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # CelebA 데이터셋은 178x218 사이즈로 정사각형이 아님
            # CelebA를 정사각형 이미지를 기대하는 UNet에 넣기 위해서는
            # CenterCrop이나 RandomCrop을 Resize 뒤에 넣어서 
            # 직사각형 이미지안에서 정사각형 조각을 잘라내는 추가 조정이 필요
            torchvision.transforms.CenterCrop(image_size),
            # TODO: 정사각형으로 전처리 되어 있는 CelebA HQ 데이터셋으로 변경하기
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        """
        Size of the dataset
        """
        return len(self._files)

    def __getitem__(self, index: int):
        """
        Get an image
        """
        img = Image.open(self._files[index])
        return self._transform(img)


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset

    labml.lab을 안쓰는 방향으로 수정
    """

    def __init__(self, root:str, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
        ])

        # datasets 폴더
        root_dir = Path(root)

        super().__init__(str(root_dir), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


def main():

    yaml_root = input("실험 설정 yaml 파일 경로를 입력하세요. : ")

    print(yaml_root)

    yaml_path = Path(yaml_root)

    if not yaml_path.exists():
        print("잘못된 경로 입니다.")
        print("실험 종료")
        return
    elif not yaml_path.is_file():
        print("파일이 아닙니다.")
        print("실험 종료")
        return

    print("".center(100, "-"))

    time_start = datetime.now()

    train_start = time_start.strftime("%Y%m%d_%H%M%S")

    # TODO: seed 고정 코드

    with open(yaml_path, "r") as f:
        loaded_dict = yaml.safe_load(f)
        # Create configurations
        configs = Configs(**loaded_dict)
    # TODO: resume 일 경우 pth 안에 저장된 yaml 값으로 바꾸기

    ddpm = DDPM(configs)

    init_end = datetime.now()
    init_time = init_end - time_start
    init_time = str(init_time).split(".")[0]
    print(f"==>> initialization_time: {init_time}")

    print("".center(100, "-"))

    ddpm.run()

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