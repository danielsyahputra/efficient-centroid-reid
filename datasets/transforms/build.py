# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
from .random_erasing import RandomErasing
from .trans_gray import LGT
from .augmentations import *

class ReidTransforms():

    def __init__(self, cfg):
        self.cfg = cfg

    def build_transforms(self, is_train=True):
        normalize_transform = T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        if is_train:
            seqs = [
                T.Resize(self.cfg.INPUT.SIZE_TRAIN),
            ]
            if self.cfg.INPUT.USE_LGT:
                seqs.append(
                    LGT(probability=self.cfg.INPUT.LGT_PROB)
                )
            
            if self.cfg.INPUT.USE_LGPR:
                seqs.append(
                    LGPR(0.4)
                )
            
            if self.cfg.INPUT.USE_GGPR:
                seqs.append(
                    T.RandomGrayscale(0.05)
                )
            
            if self.cfg.INPUT.USE_FUSE_RGB:
                seqs.append(
                    Fuse_RGB_Gray_Sketch()
                )
            
            seqs += [
                T.RandomHorizontalFlip(p=self.cfg.INPUT.PROB),
                T.Pad(self.cfg.INPUT.PADDING),
                T.RandomCrop(self.cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=self.cfg.INPUT.RE_PROB, mean=self.cfg.INPUT.PIXEL_MEAN)
            ]
            transform = T.Compose(seqs)
        else:
            transform = T.Compose([
                T.Resize(self.cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                normalize_transform
            ])

        return transform