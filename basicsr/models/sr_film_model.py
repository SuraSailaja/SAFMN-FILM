import torch
from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRFiLMModel(BaseModel):
    """
    BasicSR SRModel variant that passes augmentation metadata
    to a FiLM-conditioned generator.
    """

    def __init__(self, opt):
        super(SRFiLMModel, self).__init__(opt)

        self.net_g = build_network(
            opt['network_g']
        )
        self.net_g = self.model_to_device(
            self.net_g
        )

        self.print_network(self.net_g)

        load_path = self.opt['path'].get(
            'pretrain_network_g',
            None
        )

        if load_path is not None:
            param_key = self.opt['path'].get(
                'param_key_g',
                'params'
            )

            self.load_network(
                self.net_g,
                load_path,
                self.opt['path'].get(
                    'strict_load_g',
                    True
                ),
                param_key
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):

        self.net_g.train()

        train_opt = self.opt['train']

        self.ema_decay = train_opt.get(
            'ema_decay',
            0
        )

        if self.ema_decay > 0:

            logger = get_root_logger()

            logger.info(
                f'Use Exponential Moving Average '
                f'with decay: {self.ema_decay}'
            )

            self.net_g_ema = build_network(
                self.opt['network_g']
            ).to(self.device)

            if self.opt['path'].get(
                'pretrain_network_g',
                None
            ) is not None:

                self.load_network(
                    self.net_g_ema,
                    self.opt['path']['pretrain_network_g'],
                    self.opt['path'].get(
                        'strict_load_g',
                        True
                    ),
                    'params_ema'
                )

            else:
                self.model_ema(0)

            self.net_g_ema.eval()

        # Losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(
                train_opt['pixel_opt']
            ).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(
                train_opt['fft_opt']
            ).to(self.device)
        else:
            self.cri_fft = None

        if self.cri_pix is None and self.cri_fft is None:
            raise ValueError(
                'Pixel and frequency losses are both None.'
            )

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):

        train_opt = self.opt['train']

        optim_params = []

        for k, v in self.net_g.named_parameters():

            if v.requires_grad:
                optim_params.append(v)

            else:
                logger = get_root_logger()

                logger.warning(
                    f'Params {k} will not be optimized.'
                )

        optim_type = train_opt[
            'optim_g'
        ].pop('type')

        self.optimizer_g = self.get_optimizer(
            optim_type,
            optim_params,
            **train_opt['optim_g']
        )

        self.optimizers.append(
            self.optimizer_g
        )

    def feed_data(self, data):

        self.lq = data['lq'].to(
            self.device
        )

        self.aug_cond = data.get(
            'aug_cond',
            None
        )

        if self.aug_cond is None:
            self.aug_cond = torch.zeros(
                self.lq.size(0),
                5,
                device=self.device,
                dtype=self.lq.dtype
            )
        else:
            self.aug_cond = self.aug_cond.to(
                self.device
            ).to(self.lq.dtype)

        if 'gt' in data:
            self.gt = data['gt'].to(
                self.device
            )

    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()

        self.output = self.net_g(
            self.lq,
            self.aug_cond
        )

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:

            l_pix = self.cri_pix(
                self.output,
                self.gt
            )

            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_fft:

            l_fft = self.cri_fft(
                self.output,
                self.gt
            )

            l_total += l_fft
            loss_dict['l_freq'] = l_fft

        l_total.backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(
            loss_dict
        )

        if self.ema_decay > 0:
            self.model_ema(
                decay=self.ema_decay
            )

    def test(self):

        if hasattr(self, 'net_g_ema'):

            self.net_g_ema.eval()

            with torch.no_grad():

                self.output = self.net_g_ema(
                    self.lq,
                    self.aug_cond
                )

        else:

            self.net_g.eval()

            with torch.no_grad():

                self.output = self.net_g(
                    self.lq,
                    self.aug_cond
                )

            self.net_g.train()

    def get_current_visuals(self):

        out_dict = OrderedDict()

        out_dict['lq'] = (
            self.lq.detach().cpu()
        )

        out_dict['result'] = (
            self.output.detach().cpu()
        )

        if hasattr(self, 'gt'):
            out_dict['gt'] = (
                self.gt.detach().cpu()
            )

        return out_dict

    def save(self, epoch, current_iter):

        if hasattr(self, 'net_g_ema'):

            self.save_network(
                [
                    self.net_g,
                    self.net_g_ema
                ],
                'net_g',
                current_iter,
                param_key=[
                    'params',
                    'params_ema'
                ]
            )

        else:

            self.save_network(
                self.net_g,
                'net_g',
                current_iter
            )

        self.save_training_state(
            epoch,
            current_iter
        )
