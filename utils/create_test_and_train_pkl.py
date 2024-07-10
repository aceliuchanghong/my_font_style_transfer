import os
import pickle

from utils.deal_before_generate import resize_thin_character
from utils.judge_font import get_files
from utils.util import write_pkl

script = {
    "CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl'],
    'JAPANESE': ['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
    "ENGLISH": ['CASIA_ENGLISH', 'English_content.pkl']
}

root = '../data'
dataset = 'CHINESE'

if __name__ == '__main__':
    writer_dict_data_path = os.path.join(root, script[dataset][0])
    all_writer = pickle.load(open(os.path.join(writer_dict_data_path, 'writer_dict.pkl'), 'rb'))
    train_writer_pkl_name = []
    test_writer_pkl_name = []
    get_suffix = ".png"
    save_pics_path = 'suit_pics3'
    show_pics_num = 2

    for pkl_name in all_writer['train_writer']:
        train_writer_pkl_name.append(pkl_name.split('.')[0])
    for pkl_name in all_writer['test_writer']:
        test_writer_pkl_name.append(pkl_name.split('.')[0])
    # print(train_writer_pkl_name)
    font_path = r'D:\soft\FontForgeBuilds\LCH_pics'
    folder_names = [name for name in os.listdir(font_path) if os.path.isdir(os.path.join(font_path, name))]
    for i in range(len(folder_names)):
        print(folder_names[i], train_writer_pkl_name[i])
        save_pics_dir = save_pics_path + "/" + folder_names[i]
        # 确保保存图片的目录存在
        if not os.path.exists(save_pics_dir):
            os.makedirs(save_pics_dir)
        base_dir = font_path + '/' + folder_names[i]
        # 获取文件列表
        files_list = get_files(base_dir, get_suffix)
        # 图片骨架
        resize_thin_character(files_list, save_pics_dir, show_pics_num, save_chinese_name=True)
        # 写pkl
        img_file_list = get_files(save_pics_dir, get_suffix)
        write_pkl(file_path=save_pics_path, file_name=train_writer_pkl_name[i] + '.pkl', imgs_path=img_file_list,
                  show_pic_num=2)
