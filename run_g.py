import os
import pickle
from utils.util import fix_seed
from z_new_start.FontConfig import new_start_config
from z_new_start.FontDataset import FontDataset
from z_new_start.FontUtils import CoorsRender, _get_coors_decode
import argparse
import torch

from z_new_start.generate_utils.read_coordinates_pkl import draw_character_strokes

fix_seed(840)
#fix_seed(857)
tensor0 = torch.randn(1, 20, 200, 4)
tensor1 = torch.randn(1, 20, 200, 4)
pred = []
pred.append(tensor0)
pred.append(tensor1)
images = torch.randn(1, 12, 1, 64, 64)
gd = FontDataset(is_train=False, is_dev=False)
outputs, _ = _get_coors_decode(CoorsRender(), p=pred, images=images, dropout=new_start_config['train']['dropout'],
                               gd=gd)
print(outputs, _)


def main(opt):
    out_path = opt.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    coor = pickle.load(open(opt.pkl, 'rb'))
    try:
        del coor['font_name']
    except Exception as e:
        pass
    images = draw_character_strokes(coor, scale_factor=opt.scale, degree=opt.degree)
    for char, image in images.items():
        image.save(f"{out_path}/{char}.png")  # 保存图像
    print(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='z_new_start/generate_utils/files/coors_pics_path', help='输出图片目录')
    parser.add_argument('--pkl', default=r'D:\aProject\py\my_font_style_transfer\Saved\samples\new.pkl',
                        help='读取文件')
    parser.add_argument('--scale', default=0.27, type=float, help='图片缩放尺寸')
    parser.add_argument('--degree', default=0.00, type=float)
    opt = parser.parse_args()
    main(opt)
