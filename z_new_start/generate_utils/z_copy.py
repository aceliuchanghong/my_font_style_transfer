import os
import shutil

from utils.judge_font import get_files

path_basic = r'D:\aProject\py\my_font_style_transfer\z_new_start\generate_utils\files\coors_pics_path'
coor_pkl = r'D:\download\font_coor\lch_coor'
img_pkl = r'D:\download\font_pics_pkl\lch_pics_pkl'
coor_pkl_new = r'D:\download\new_font\coors'
img_pkl_new = r'D:\download\new_font\img'

name_list = get_files(path_basic, '.png')
for name in name_list:
    name = os.path.basename(name).split('.')[0]
    old_coor_name = os.path.join(coor_pkl, name + '.pkl')
    old_img_name = os.path.join(img_pkl, name + '.pkl')
    if os.path.exists(old_coor_name) and os.path.exists(old_img_name):
        shutil.copy(old_coor_name, coor_pkl_new)
        shutil.copy(old_img_name, img_pkl_new)
        print(old_coor_name + " move to " + coor_pkl_new)
        print(old_img_name + " move to " + img_pkl_new)
        print()
