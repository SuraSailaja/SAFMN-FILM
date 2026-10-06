import cv2
import numpy as np
import random

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file
)
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedHeavyAugDataset(data.Dataset):
    """Paired dataset with exactly one blind augmentation per sample.

    Choices:
        1. horizontal flip
        2. rotation
        3. Gaussian blur on LR only
        4. Gaussian noise on LR only

    The augmentation type is NOT provided to the model.
    """

    def __init__(self, opt):
        super(PairedHeavyAugDataset, self).__init__()

        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [
                self.lq_folder, self.gt_folder
            ]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt']
            )

        elif 'meta_info_file' in opt and opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt'],
                opt['meta_info_file'],
                self.filename_tmpl
            )

        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt'],
                self.filename_tmpl
            )

    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'),
                **self.io_backend_opt
            )

        scale = self.opt['scale']

        # Load GT
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # Load LR
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Training crop
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            img_gt, img_lq = paired_random_crop(
                img_gt,
                img_lq,
                gt_size,
                scale,
                gt_path
            )

            # Exactly ONE augmentation.
            aug_type = random.choice([
                'hflip',
                'rotation',
                'blur',
                'noise'
            ])

            if aug_type == 'hflip':
                img_gt = np.flip(img_gt, axis=1).copy()
                img_lq = np.flip(img_lq, axis=1).copy()

            elif aug_type == 'rotation':
                k = random.choice([1, 2, 3])
                img_gt = np.rot90(img_gt, k=k).copy()
                img_lq = np.rot90(img_lq, k=k).copy()

            elif aug_type == 'blur':
                sigma = random.uniform(0.5, 1.5)

                kernel_size = int(
                    2 * round(3 * sigma) + 1
                )

                img_lq = cv2.GaussianBlur(
                    img_lq,
                    (kernel_size, kernel_size),
                    sigmaX=sigma,
                    sigmaY=sigma
                )

            elif aug_type == 'noise':
                sigma = random.uniform(2.0, 10.0) / 255.0

                noise = np.random.normal(
                    0.0,
                    sigma,
                    img_lq.shape
                ).astype(np.float32)

                img_lq = np.clip(
                    img_lq + noise,
                    0.0,
                    1.0
                )

        else:
            aug_type = 'none'

            # Validation/test crop.
            img_gt = img_gt[
                0:img_lq.shape[0] * scale,
                0:img_lq.shape[1] * scale,
                :
            ]

        # Color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(
                img_gt,
                y_only=True
            )[..., None]

            img_lq = rgb2ycbcr(
                img_lq,
                y_only=True
            )[..., None]

        # BGR -> RGB, HWC -> CHW
        img_gt, img_lq = img2tensor(
            [img_gt, img_lq],
            bgr2rgb=True,
            float32=True
        )

        # Normalize
        if self.mean is not None or self.std is not None:
            normalize(
                img_lq,
                self.mean,
                self.std,
                inplace=True
            )
            normalize(
                img_gt,
                self.mean,
                self.std,
                inplace=True
            )

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
