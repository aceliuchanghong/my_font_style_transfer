import argparse
import logging
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from z_new_start.FontModel import FontModel
from z_new_start.FontConfig import new_start_config
from z_new_start.FontDataset import FontDataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.manual_seed(random_seed)


def main(opt):
    conf = new_start_config
    train_conf = conf['train']
    if opt.dev:
        data_conf = conf['dev']
    else:
        data_conf = conf['test']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 禁用 cuDNN autotuner
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 强制禁用 cuDNN 后端
    torch.backends.cudnn.enabled = False

    fix_seed(train_conf['seed'])
    logger.info(f"seed: {train_conf['seed']}")

    train_dataset = FontDataset(is_train=True, is_dev=opt.dev)
    valid_dataset = FontDataset(is_train=False, is_dev=opt.dev)
    logger.info(
        f"\nThe number of training images:  {len(train_dataset)}\nThe number of valid images: {len(valid_dataset)}"
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=data_conf['PER_BATCH'],
                              shuffle=True,
                              drop_last=False,
                              collate_fn=train_dataset.collect_function,
                              num_workers=data_conf['NUM_THREADS'],
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=data_conf['PER_BATCH'],
                              shuffle=True,
                              drop_last=False,
                              collate_fn=valid_dataset.collect_function,
                              num_workers=data_conf['NUM_THREADS'],
                              pin_memory=True)
    model = FontModel(
        d_model=train_conf['d_model'],
        num_head=train_conf['num_head'],
        num_encoder_layers=train_conf['num_encoder_layers'],
        num_glyph_encoder_layers=train_conf['num_glyph_encoder_layers'],
        num_gly_decoder_layers=train_conf['num_gly_decoder_layers'],
        dim_feedforward=train_conf['dim_feedforward'],
        dropout=train_conf['dropout'],
        activation="relu",
        normalize_before=True,
        return_intermediate_dec=True,
        train_conf=train_conf,
    )
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    elif torch.cuda.is_available():
        logger.info("Using single GPU")
    else:
        logger.info("Using CPU")

    from torchsummary import summary
    summary(model, input_size=[(12, 1, 64, 64), (20, 200, 4), (1, 64, 64)], device=device)


if __name__ == '__main__':
    """
    python z_new_start/FontTest.py
    python z_new_start/FontTest.py --dev
    python z_new_start/FontTest.py --pretrained_model xx/xx.pth --dev
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', default='', help='pre-trained model')
    parser.add_argument('--dev', action='store_true', help='加--dev则opt.dev=True为生产环境')
    opt = parser.parse_args()
    main(opt)
