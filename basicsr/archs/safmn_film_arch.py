import torch
import torch.nn as nn

from basicsr.archs.safmn_arch import LayerNorm, SAFM, CCM
from basicsr.utils.registry import ARCH_REGISTRY


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.

    y = gamma * x + beta

    The implementation follows the core FiLM operation used by
    Perez et al.: feature-wise scale followed by feature-wise shift.
    """

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class FiLMGenerator(nn.Module):
    """
    Generate per-channel gamma and beta from augmentation metadata.

    cond_dim = 5:
        [hflip, rotation, blur, noise, severity]

    The final layer is initialized so that the initial FiLM operation
    is approximately identity: gamma=1, beta=0.
    """

    def __init__(self, cond_dim=5, dim=36, hidden_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * dim)
        )

        # Start from identity modulation:
        # gamma = 1, beta = 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        with torch.no_grad():
            self.net[-1].bias[:dim].fill_(1.0)

    def forward(self, cond):
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta


class FiLMAttBlock(nn.Module):
    """
    SAFMN attention block with FiLM conditioning.

    Original SAFMN:
        x = SAFM(LN(x)) + x
        x = CCM(LN(x)) + x

    D:
        x = SAFM(FiLM(LN(x), condition)) + x
        x = CCM(LN(x)) + x
    """

    def __init__(self, dim, ffn_scale=2.0,
                 cond_dim=5, film_hidden=32):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.film_generator = FiLMGenerator(
            cond_dim=cond_dim,
            dim=dim,
            hidden_dim=film_hidden
        )
        self.film = FiLM()

        self.safm = SAFM(dim)
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x, cond):
        # Generate condition-specific gamma/beta
        gamma, beta = self.film_generator(cond)

        # Condition the features before SAFM
        x = self.safm(
            self.film(self.norm1(x), gamma, beta)
        ) + x

        # Keep the original CCM path unchanged
        x = self.ccm(self.norm2(x)) + x

        return x


@ARCH_REGISTRY.register()
class SAFMNFiLM(nn.Module):
    """
    SAFMN + FiLM conditioning for augmentation-aware SR.

    cond:
        [B, 5]
        [hflip, rotation, blur, noise, severity]

    For clean validation/test images, cond should be all zeros.
    Because FiLM is initialized with gamma=1 and beta=0, zero
    conditioning starts as an identity transformation.
    """

    def __init__(self,
                 dim=36,
                 n_blocks=8,
                 ffn_scale=2.0,
                 upscaling_factor=4,
                 cond_dim=5,
                 film_hidden=32):
        super().__init__()

        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.ModuleList([
            FiLMAttBlock(
                dim=dim,
                ffn_scale=ffn_scale,
                cond_dim=cond_dim,
                film_hidden=film_hidden
            )
            for _ in range(n_blocks)
        ])

        self.to_img = nn.Sequential(
            nn.Conv2d(
                dim,
                3 * upscaling_factor ** 2,
                3, 1, 1
            ),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x, cond=None):
        x = self.to_feat(x)

        if cond is None:
            cond = torch.zeros(
                x.size(0),
                5,
                device=x.device,
                dtype=x.dtype
            )

        for block in self.feats:
            x = block(x, cond)

        x = x + self.to_feat_output(x) if False else x

        x = self.to_img(x)
        return x


if __name__ == '__main__':
    model = SAFMNFiLM(
        dim=36,
        n_blocks=8,
        ffn_scale=2.0,
        upscaling_factor=4,
        cond_dim=5,
        film_hidden=32
    )

    x = torch.randn(2, 3, 64, 64)
    cond = torch.randn(2, 5)

    y = model(x, cond)

    print(model)
    print('Input:', x.shape)
    print('Condition:', cond.shape)
    print('Output:', y.shape)
    print('Parameters:',
          sum(p.numel() for p in model.parameters()))
