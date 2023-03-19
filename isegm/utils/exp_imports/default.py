import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from app.ClickSEG.isegm.data.datasets import *
from app.ClickSEG.isegm.model.losses import *
from app.ClickSEG.isegm.data.transforms import *
from app.ClickSEG.isegm.model.metrics import AdaptiveIoU
from app.ClickSEG.isegm.data.points_sampler import MultiPointSampler
from app.ClickSEG.isegm.utils.log import logger
from app.ClickSEG.isegm.model import initializer

from app.ClickSEG.isegm.model.is_hrnet_model import HRNetModel
from app.ClickSEG.isegm.model.is_deeplab_model import DeeplabModel
from app.ClickSEG.isegm.model.is_segformer_model import SegFormerModel
from app.ClickSEG.isegm.model.is_strong_baseline import BaselineModel