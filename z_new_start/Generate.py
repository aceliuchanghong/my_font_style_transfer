import argparse
import logging
import pickle
import torch
import sys
import os
from torch.utils.data import DataLoader
import numpy as np

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from z_new_start.FontModel import FontModel
from z_new_start.FontConfig import new_start_config
from z_new_start.generate_utils.gen_char_img_pkl import get_files
from utils.util import write_pkl
from z_new_start.FontDataset import FontDataset
from z_new_start.z_train import fix_seed

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    img_path_list, style_batch_data = get_files(data_conf['style_img_path'], data_conf['suffix']), []
    if len(img_path_list) < train_conf['style_img_num']:
        logger.error(
            f"not enough images, at least {train_conf['style_img_num']} {data_conf['suffix']}s in {data_conf['style_img_path']}"
        )
        return
    style_samples = write_pkl(data_conf['style_pkl_file_path'], 'generate.pkl',
                              img_path_list[:train_conf['style_img_num']])

    new_dic = pickle.load(open(os.path.join(data_conf['style_pkl_file_path'], 'generate.pkl'), 'rb'))
    try:
        seed = int(np.sum(new_dic[0]['img']) / 1000)
    except Exception as e:
        seed = train_conf['seed']
    fix_seed(seed)
    logger.info(f"seed: {seed}")

    generate_dataset = FontDataset(is_train=opt.dev, is_dev=opt.dev)
    generate_loader = DataLoader(generate_dataset, 1, True,
                                 drop_last=False,
                                 collate_fn=generate_dataset.collect_function,
                                 num_workers=data_conf['NUM_THREADS'],
                                 pin_memory=True)
    for i, img in enumerate(style_samples):
        img = img['img'] / 255.0
        char_img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        style_batch_data.append(char_img_tensor)
    images = torch.stack([item for item in style_batch_data]).unsqueeze(0)
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
    model.to(device)
    if len(opt.pretrained_model) > 0:
        state_dict = torch.load(opt.pretrained_model)
        model_state_dict = state_dict['model_state_dict']
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        logger.info('loaded pretrained model from {}'.format(opt.pretrained_model))
    if isinstance(model, torch.nn.DataParallel):
        coors_path = model.module.inference(images, generate_dataset, generate_loader)
    else:
        coors_path = model.inference(images, generate_dataset, generate_loader)
    logger.info('result coordinates path:{}'.format(coors_path))


if __name__ == '__main__':
    """
    python z_new_start/Generate.py
    python z_new_start/Generate.py --dev
    python z_new_start/Generate.py --dev --pretrained_model xx.pt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model',
                        default='/mnt/data/llch/my_font_style_transfer/z_new_start/save_model/best_model.pt',
                        help='pre-trained model'
                        )
    parser.add_argument('--dev', action='store_true', help='加--dev则opt.dev=True为生产环境')
    opt = parser.parse_args()
    main(opt)
