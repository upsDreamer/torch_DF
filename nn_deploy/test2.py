import torch
import torch.fx
from torch import Tensor, nn
from typing import Callable, Iterable, List, Optional, Tuple, Union
import numpy as np

class Mask(nn.Module):
    def __init__(self, post_filter: bool = False, eps: float = 1e-12):
        super().__init__()
        # self.erb_inv_fb: Tensor
        # self.register_buffer("erb_inv_fb", erb_inv_fb)
        self.clamp_tensor = torch.__version__ > "1.9.0" or torch.__version__ == "1.9.0"
        self.post_filter = post_filter
        self.eps = eps

    def pf(self, mask: Tensor, beta: float = 0.02) -> Tensor:
        """Post-Filter proposed by Valin et al. [1].

        Args:
            mask (Tensor): Real valued mask, typically of shape [B, C, T, F].
            beta: Global gain factor.
        Refs:
            [1]: Valin et al.: A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech.
        """
        mask_sin = mask * torch.sin(np.pi * mask / 2)
        mask_pf = (1 + beta) * mask / (1 + beta * mask.div(mask_sin.clamp_min(self.eps)).pow(2))
        return mask_pf
    
    def forward(self, spec: Tensor, mask: Tensor, atten_lim: Optional[Tensor] = None) -> Tensor:
        # spec (real) [B, 1, T, F, 2], F: freq_bins
        # mask (real): [B, 1, T, Fe], Fe: erb_bins
        # atten_lim: [B]
        if not self.training and self.post_filter:
            mask = self.pf(mask)
        if atten_lim is not None:
            # dB to amplitude
            atten_lim = 10 ** (-atten_lim / 20)
            # Greater equal (__ge__) not implemented for TorchVersion.
            if self.clamp_tensor:
                # Supported by torch >= 1.9
                mask = mask.clamp(min=atten_lim.view(-1, 1, 1, 1))
            else:
                m_out = []
                for i in range(atten_lim.shape[0]):
                    m_out.append(mask[i].clamp_min(atten_lim[i].item()))
                mask = torch.stack(m_out, dim=0)
        # mask = mask.matmul(self.erb_inv_fb)  # [B, 1, T, F]
        #if not spec.is_complex():
        mask = mask.unsqueeze(4)
        return spec * mask


class DfOutputReshapeMF(nn.Module):
    """Coefficients output reshape for multiframe/MultiFrameModule

    Requires input of shape B, C, T, F, 2.
    """

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        # new_shape = list(coefs.shape)
        new_shape = torch.Tensor(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs

# model = Mask()
model = DfOutputReshapeMF(1,2)
traced = torch.fx.symbolic_trace(model)
print(traced)
traced.print_readable()