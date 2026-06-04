"""
출처: https://nn.labml.ai/diffusion/ddpm/unet.html
"""

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn


class Swish(nn.Module):
    """
    Swish activation function

    x * sigmoid(x)
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    transfomer의 positional encoding처럼 sine, cosine 사용하고
    문장의 position 부분만 time step으로 변경
    https://nn.labml.ai/transformers/positional_encoding.html 참고
    """

    def __init__(self, n_channels: int):
        """
        n_channels : embedding의 최종 출력 차원
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # 
        # t.size() = [batch_size]
        # t의 각 원소들은 1~T 사이의 임의 추출된 값으로 해당 batch에서 복원을 수행할 time-step t값
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # d == half_dim
        half_dim = self.n_channels // 8
        # 여기서  self.n_channels // 8인 sin, cos 생성해야 concat 후에
        # emb 다음 linear 층 입력 차원 수인 self.n_channels // 4가 된다.

        # 10000^(i/(d-1)) == e^(log10000 * i /(d-1))
        emb = math.log(10_000) / (half_dim - 1)
        # i/(d-1): i ∈ [0, d) 이므로 분모도 d-1로 두어서 
        # i가 최대일때 분수 값이 1이 되도록 설정

        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        # emb.size() == [half_dim]
        # 1/(10000^(...)) 꼴이므로 지수에 * -1 되어 있음 

        emb = t[:, None] * emb[None, :]
        # t[:, None]은 [batch_size, 1]
        # emb[None, :]은 [1, d]
        # ==> [batch_size, d]

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 트랜스포머의 PE와 다르게 sine, cosine 홀짝 교차를 하지 않는다.
        # 이 emb를 생성한 바로 뒤에 nn.Linear 레이어를 통과하므로 linear 층의 가중치 각 행들과 emb의 내적 계산을 한다.
        # 이 때 각 행의 내적을 보면 (w_i1, w_i2, w_i3, w_i4, ...) · (sin, cos, sin, cos, ....)를 하는 것과
        # (w_i1, w_i3, w_i5, w_i7, ..., w_i2, w_i4, w_i6, ...) · (sin, sin, sin, sin, ...., cos, cos, cos, ....)로 계산하는 것은 결과가 완전히 동일하다.
        # 따라서 홀짝교차보다 단순 cat이 메모리 연속성 + 연산 속도 빠름 + 코드 간결 등 이점이 많으므로 홀짝 교차를 하지 않는다.
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        # emb.size() == [batch_size, 2d]
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        # [batch_size, 2d] 에서 [batch_size, 8d == self.n_channels]로 출력 차원수 증가
        emb = self.lin2(emb)

        return emb


class ResidualBlock(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number of channels in the time step (t) embeddings
        * `n_groups` is the number of groups for group normalization (https://nn.labml.ai/normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        # conv 통과 후에 이미지 width, height는 불변
        # input_length - kernel_size(3) + 1 + (2 * pad_size(1)) = input_length

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # h는 [batch_size, out_channels, height, width]
        
        # Add time embeddings
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        # t는 [batch_size, time_channels]
        # self.time_emb(self.time_act(t))는 [batch_size, out_channels]
        # [:, :, None, None]을 하면
        # [batch_size, out_channels, 1, 1]이 되어 h에 broadcasting이 되면서 더해질 수 있다.

        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    ### Attention block

    transformer MHA와 유사한 형태(https://nn.labml.ai/transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
            DDPM 원 논문과 nn.labml.ai 코드 구현은 n_heads == 1로 고정
                @@@ nn.labml.ai 코드는 한개의 해상도에서만 attention block을 사용한 DDPM 원 논문과 다르게
                @@@ Improved Denoising Diffusion Probabilistic Models 논문의 개선된 모델처럼 
                @@@ attention block을 두개의 해상도 영역에서 사용한다. 
                @@@ 그렇지만 Improved Denoising Diffusion Probabilistic Models 의 개선된 모델처럼 head를 4로 쓰지 않고 있다. 
                @@@ TODO: ==> n_heads == 4로 변경해볼 것
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for group normalization (https://nn.labml.ai/normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)

        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # 헤드 별로 q, k, v 생성 linear를 따로따로 만드는 경우 nn.Linear(n_channels, d_k)가 n_heads * 3개가 생기지만
        # nn.Linear(n_channels, n_heads * d_k * 3) 하나만 사용해서 결과물을 head별, q, k, v별로 쪼개주는 방식이 더 효율적
            # 이미지 인풋 [batch_size, n_channels, height, width]을 attention 연산을 하기 위해 공간 2차원을 1차원 seq로 변경
            # ==> [batch_size, n_channels, height*width] (height*width == seq ==> seq 하나하나는 픽셀 하나)
            # seq와 channel 축 transpose ==> [batch_size, seq, n_channels]
            # self.projection에 입력해 head 별 q, k, v 생성 (q, k, v 모두 d_k 차원)
            # ==> [batch_size, seq, n_heads * d_k * 3]
            # view로 [batch_size, seq, n_heads, d_k * 3]로 변경
            # chunk로 [batch_size, seq, n_heads, d_k]인 q, k, v로 분해
            # ([batch_size, n_heads, seq, d_k]로 permute 안하고 einsum으로 attention score 연산 해결)


        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)

        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 총 b*h*d개인 (b, h, d) 순서쌍 하나 마다 
        # q_(b, i, h, d) * k_(b, j, h, d)이 i*j번 곱해진 후 (b, i, j, h, d) 총 b*h*d*i*j개 (q_id * k_jd)
        # 여기서 sum over d를 해서 (b, i, j, h) 총 b*i*j*h개 항 (∑_d (q_id * k_jd) == Q@K^T)
            # q (b1, i, h1, d1), k (b2, j, h2, d2)로 b, h, d가 중복 등장하지 않을 경우, 
            # q 원소 각각에 k의 원소 전부가 곱해지므로(outer(tensor) product) b1 * i * h1 * d1 * b2 * j * h2 * d2 개의 원소가 있지만,
            # b, h, d로 짝을 이루면 b, h, d가 같은 원소들끼리만 곱을 하므로 q의 원소 하나당 k의 원소 j개가 곱해진다.
            # q의 원소가 총 b * i * h * d 개 이므로 곱해진 결과에는 b * i * j * h * d 개의 원소가 있다.
            # 여기서 출력에서 생략된 d 차원을 따라서 summation을 진행하면 b * i * j * h 개의 원소가 남는다.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # torch.einsum : 지정된 차원을 따라 입력 피연산자(operands) 요소들의 곱의 합을 구한다.
        # torch.einsum 규칙 1) 입력(bihd,bjhd)에만 있고 출력(bijh)에 없는 인덱스(d == d_k)는 해당 인덱스(d) 차원을 따라서 모두 더한다(sum over d).
        # torch.einsum 규칙 2) 여러 입력 텐서에서 중복해서 등장하는 인덱스(b,h,d)은 그 인덱스를 기준으로 짝을 맞추어 원소들끼리 곱한다.
            # @@@ 입력 텐서들에 중복 등장하는 인덱스가 없어도 입력 텐서 요소들의 곱(element-wise 곱)은 무조건 진행(밑의 outer product 예시 확인)
        # ex: torch.einsum('ij,ij->ij', A, B) == 아다마르 곱(Matrix Element-wise Multiplication)
        # ex: torch.einsum('ij,jk->ik', A, B) == 행렬 곱
        # ex: torch.einsum('i,j->ij', a, b) == outer product
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # dim=1은 query, dim=2는 key

        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # 입력 텐서 양쪽 모두에 b, j, h 등장
        # (b, j, h) 순서쌍 하나 마다 i*d번 곱셈 (a_ij * v_jd)
        # 그 다음 sum over j (∑_j (a_ij * v_jd) == Attn@V)

        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)

        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x
        # x는 [batch_size, seq, n_channels] 형태로 위에서 변경된 상태이므로 res에 바로 더해질 수 있음

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        # res는 결국 각 픽셀마다 그 픽셀이 그림 내의 모든 픽셀과 어떤 관계(attention score)에 있는지 계산한 후 
        # 그 attention score를 이용해 그 픽셀과 연관이 큰 픽셀(자기 자신 포함) 신호들을 합한 결과를 얻는다.
        # 여기에 최종적으로 원본 x가 더해져서 출력된다. (원본 x에 더해지기 전에 mha 결과는 linear(self.output) 층 한번 통과)
        return res


class DownBlock(nn.Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Downsample(nn.Module):
    """
    ### Scale down the feature map by 1/2 
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))
        # out_len = 1 + 버림((in_len + 2 * padding - kernel_size)/stride)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Upsample(nn.Module):
    """
    ### Scale up the feature map by 2
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))
        # out_len = (in_len - 1) * stride - 2 * padding + kernel_size

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    """
    ## U-Net
    @@@ 여기서 구현되는 UNet은 원본 UNet보다는 PixelCNN++에 기반한 Wide ResNet 구조에 가깝다
    @@@ 따라서 각 해상도에서 한번씩만 concat을 하는 구조가 아니라
    @@@ 전반부의 모든 모듈 결과 피처 맵들을 저장해
    @@@ 후반부에서 그 모든 피처맵에 대해 concat을 진행한다.
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True), # 고 해상도일 때는 연산이 비싼 attention 미사용 @@@ DDPM 원 논문은 attention을 한개의 해상도에서만 사용
                 n_blocks: int = 2):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)
        # [batch_size, time_channels == n_channels * 4]
        # ch_mults가 (1, 2, 2, 4)로 최대 n_channels * 4 까지 채널 수가 늘어나므로 
        # TimeEmbedding은 최대치(n_channels * 4)에 맞추어 최대한 복잡하고 풍부한 시간 정보를 가지는 고차원 벡터로 변환해두고
        # 각 층에서 linear를 한번 더 통과 시켜 각자의 채널 수에 맞게 채널 수를 조정한다.

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # out_channels = in_channels * ch_mults[i]
            # 26.06.05 기준 https://nn.labml.ai/diffusion/ddpm/unet.html 코드 오류
                # 누적 곱을 하면 최종 채널 수가 n_channels * 4 가 아니라 n_channels * 16이 되버린다.
            # in_channels이 아니라 n_channels이어야 최종 채널 수가 time embedding 채널 수와 같아진다.
            out_channels = n_channels * ch_mults[i]
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # ch_mults == (1, 2, 2, 4)

            # Add `n_blocks`
                # n_blocks == 2
            for _ in range(n_blocks):
                # 각 해상도마다 DownBlock을 두번 반복
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                # 첫번째 DownBlock은 in_channels에서 out_channels( = n_channels * ch_mults[i])로 채널 수가 증가하고
                in_channels = out_channels
                # 두번째 DownBlock은 out_channels으로 채널 수가 고정된 상태에서 연산

            # Down sample at all resolutions except the last
            if i < n_resolutions - 1: # i가 0, 1, 2일 때 Downsample => 0->1, 1->2, 2->3
                down.append(Downsample(in_channels))
            # @@@ down과 middle 사이에는 down sample 없음?

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )
        # residual + attention + residual
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # MiddleBlock의 두번째 인자는 출력의 채널 수가 아니라 time embedding 채널
        # ==> MiddleBlock의 입력과 출력의 채널 수는 첫번째 인자 값(out_channels)으로 고정
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # out_channels은 현재 n_channels * 4 == 256
            # @@@ 위의 코드 오류 부분을 그대로 둘 경우 n_channels * 16 == 1024

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # out_channels = in_channels // ch_mults[i]
            # 26.06.05 기준 https://nn.labml.ai/diffusion/ddpm/unet.html 코드 오류
            # 전반부의 누적 곱을 여기서 누적 나눗셈을 해서 코드 자체는 정상적으로 돌아간다.
            # 전반부의 누적 곱을 지우면 이 곳도 수정 필수
            if i > 0:
                out_channels = n_channels * ch_mults[i-1]
                # TODO: 채널 스택 방식으로 바꾸기
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # 각 해상도에서 DownBlock이 두번만 있었는데 UpBlock은 3번 있는 이유
                # 원본 UNet
                    # 전반부의 각 해상도에서 Conv 연산을 2번하고 다운샘플링 직전의 피처 맵 한개만 저장해서 
                    # 후반부 해당 해상도에서 저장한 피처 맵을 concat 한번만 진행
                # DDPM UNet (PixelCNN++에 기반한 Wide ResNet 구조)
                    # 전반부의 각 해상도에서 DownBlock을 거칠 때마다 결과 피처 맵을 저장하고, Downsample을 거친 결과 피처맵도 저장해서 총 3개의 피처 맵이 저장된다.
                    # 이 저장된 3개의 피처 맵을 후반부 해당 해상도에서 각각의 UpBlock에 같이 입력해서 concat 후 연산을 진행하므로 UpBlock이 3개 필요하다.
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


            # Up sample at all resolutions except last
            if i > 0: # i가 3, 2, 1일 때 Upsample 추가 => 3->2, 2->1, 1->0 3번.
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)
        # RGB 3채널에서 n_channels 64채널로 채널 수 증가

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                # x와 s 모두 [batch_size, channels, height, width] 형태 => channels(dim=1) 기준으로 concat해서 channel 개수 증가
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))