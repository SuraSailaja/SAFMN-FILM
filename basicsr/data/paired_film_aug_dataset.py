import random

import cv2
import numpy as np
import torch
from torch.utils import data

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file
)
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedFiLMAugDataset(data.Dataset):
    """
    Paired SR dataset for augmentation-conditioned SAFMN.

    Exactly ONE augmentation is selected per training sample:

        0 = horizontal flip
        1 = rotation
        2 = Gaussian blur
        3 = Gaussian noise

    Conditioning vector:

        [hflip, rotation, blur, noise, severity]

    HR stays clean for blur/noise.
    HR and LR are transformed together for flip/rotation.

    For validation/test:
        no augmentation
        condition = [0, 0, 0, 0, 0]
    """

    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        self.filename_tmpl = opt.get(
            'filename_tmpl', '{}'
        )

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [
                self.lq_folder,
                self.gt_folder
            ]
            self.io_backend_opt['client_keys'] = [
                'lq',
                'gt'
            ]

            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt']
            )

        elif (
            'meta_info_file' in opt
            and opt['meta_info_file'] is not None
        ):
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

    @staticmethod
    def _make_condition(aug_type, severity):
        cond = np.zeros(5, dtype=np.float32)

        # One-hot augmentation type
        cond[aug_type] = 1.0

        # Normalized severity
        cond[4] = float(severity)

        return torch.from_numpy(cond)

    @staticmethod
    def _rotate(img, k):
        return np.rot90(img, k=k).copy()

    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'),
                **self.io_backend_opt
            )

        scale = self.opt['scale']

        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        img_bytes = self.file_client.get(
            gt_path, 'gt'
        )
        img_gt = imfrombytes(
            img_bytes,
            float32=True
        )

        img_bytes = self.file_client.get(
            lq_path, 'lq'
        )
        img_lq = imfrombytes(
            img_bytes,
            float32=True
        )

        if self.opt['phase'] == 'train':

            gt_size = self.opt['gt_size']

            img_gt, img_lq = paired_random_crop(
                img_gt,
                img_lq,
                gt_size,
                scale,
                gt_path
            )

            # Exactly ONE augmentation
            aug_type = random.randrange(4)

            if aug_type == 0:
                # -------------------------
                # Horizontal flip
                # -------------------------
                img_gt = img_gt[:, ::-1, :].copy()
                img_lq = img_lq[:, ::-1, :].copy()

                severity = 1.0

            elif aug_type == 1:
                # -------------------------
                # Rotation: 90 / 180 / 270
                # -------------------------
                k = random.randint(1, 3)

                img_gt = self._rotate(img_gt, k)
                img_lq = self._rotate(img_lq, k)

                # 90 -> 1/3
                # 180 -> 2/3
                # 270 -> 1
                severity = k / 3.0

            elif aug_type == 2:
                # -------------------------
                # Gaussian blur on LR ONLY
                # -------------------------
                sigma = random.uniform(0.5, 1.5)

                kernel_size = (
                    int(round(6 * sigma + 1)) | 1
                )

                img_lq = cv2.GaussianBlur(
                    img_lq,
                    (kernel_size, kernel_size),
                    sigmaX=sigma,
                    sigmaY=sigma
                )

                # sigma 0.5 -> 0
                # sigma 1.5 -> 1
                severity = (sigma - 0.5) / 1.0

            else:
                # -------------------------
                # Gaussian noise on LR ONLY
                # -------------------------
                sigma = random.uniform(2.0, 10.0)

                noise = np.random.normal(
                    0.0,
                    sigma / 255.0,
                    img_lq.shape
                ).astype(np.float32)

                img_lq = np.clip(
                    img_lq + noise,
                    0.0,
                    1.0
                )

                # sigma 2 -> 0
                # sigma 10 -> 1
                severity = (sigma - 2.0) / 8.0

            aug_cond = self._make_condition(
                aug_type,
                severity
            )

        else:
            # Clean validation/test
            aug_cond = torch.zeros(
                5,
                dtype=torch.float32
            )

            img_gt = img_gt[
                :img_lq.shape[0] * scale,
                :img_lq.shape[1] * scale,
                :
            ]

        img_gt, img_lq = img2tensor(
            [img_gt, img_lq],
            bgr2rgb=True,
            float32=True
        )

        return {
            'lq': img_lq,
            'gt': img_gt,
            'aug_cond': aug_cond,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
