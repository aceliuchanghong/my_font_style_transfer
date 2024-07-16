import argparse
import logging
import pickle
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
from z_new_start.generate_utils.gen_char_img_pkl import get_files
from utils.util import write_pkl

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

    img_path_list = get_files(data_conf['style_img_path'], data_conf['suffix'])
    style_samples = write_pkl(data_conf['save_pkl_file_path'], 'generate.pkl', img_path_list)
    coors_std = pickle.load(open(data_conf['coors_pkl_path'], 'rb'))
    img_std = pickle.load(open(data_conf['content_pkl_path'], 'rb'))

    bs, nums, batch_data = 1, 12, []
    for i, img in enumerate(style_samples):
        if i >= nums:
            break
        img = img['img'] / 255.0
        char_img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        batch_data.append(char_img_tensor)
    image = torch.stack([item for item in batch_data]).unsqueeze(0)
    logger.info(f'image:{image.shape}')

    del coors_std['font_name']
    batch_coors_std = []
    for i, coor in enumerate(coors_std):
        if i >= nums:
            break
        std_coors_tensor = torch.zeros((20, 200, 4), dtype=torch.float32)
        for m, stroke in enumerate(list(coors_std.values())[i]):
            if m >= 20:
                break
            for n, point in enumerate(stroke):
                if n >= 200:
                    break
                std_coors_tensor[m, n] = torch.tensor(point, dtype=torch.float32)
        batch_coors_std.append(std_coors_tensor)
    coors_std = torch.stack([item for item in batch_coors_std])
    logger.info(f'coors_std:{coors_std.shape}')

    img_std_list = []
    for i, img in enumerate(img_std):
        if i >= nums:
            break
        img = img['img'] / 255.0
        char_img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        img_std_list.append(char_img_tensor)
    char_img_gt = torch.stack([item for item in img_std_list])
    logger.info(f'char_img_gt:{char_img_gt.shape}')

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

    if len(opt.pretrained_model) > 0:
        state_dict = torch.load(opt.pretrained_model)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            pass
            # model.load_state_dict(state_dict)
        logger.info('loaded pretrained model from {}'.format(opt.pretrained_model))
    model.to(device)
    model.eval()

    logger.info(f"start generating...")
    # with torch.no_grad():
    #     pred_sequence = model(image, std_coors, char_img_gt)
    #     pred_sequence = pred_sequence.cpu().numpy()
    pred_sequence = 0
    return pred_sequence


if __name__ == '__main__':
    """
    python z_new_start/FontGenerate.py
    python z_new_start/FontGenerate.py --dev
    python z_new_start/FontGenerate.py --pretrained_model xx/xx.pth --dev
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model',
                        default=r'D:\aProject\py\my_font_style_transfer\z_new_start\save_model\best_model.pt',
                        help='pre-trained model')
    parser.add_argument('--dev', action='store_true', help='加--dev则opt.dev=True为生产环境')
    opt = parser.parse_args()
    main(opt)
