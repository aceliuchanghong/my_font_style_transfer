from torch.utils.data import Dataset
import os
import pickle
import torch
from utils.judge_font import get_files
from z_new_start.FontConfig import new_start_config
import random
import numpy as np


class FontDataset(Dataset):
    def __init__(self, is_train=False, is_dev=True, train_percent=0.9):
        """
        is_train 给训练数据集还是测试数据集
        is_dev 在真正环境跑还是测试环境跑
        """
        if is_dev:
            self.config_set = 'dev'
        else:
            self.config_set = 'test'
        self.config = new_start_config

        self.img_std = pickle.load(open(self.config[self.config_set]['content_pkl_path'], 'rb'))
        self.character_std = pickle.load(open(self.config[self.config_set]['character_pkl_path'], 'rb'))
        self.coors_std = pickle.load(open(self.config[self.config_set]['coors_pkl_path'], 'rb'))
        self.pic_path = self.config[self.config_set]['z_pic_pkl_path']
        self.coordinate_path = self.config[self.config_set]['z_coordinate_pkl_path']
        self.max_stroke = self.config['train']['max_stroke']
        self.max_per_stroke_point = self.config['train']['max_per_stroke_point']
        self.num_img = self.config['train']['style_img_num']
        self.suffix = '.pkl'

        coors_pkl_list_all = get_files(self.coordinate_path, self.suffix)
        pics_pkl_list_all = get_files(self.pic_path, self.suffix)

        self.can_be_used_font = []
        for i, font_pic_pkl in enumerate(pics_pkl_list_all):
            font_name = os.path.basename(font_pic_pkl).split('.')[0]
            for coors_pkl in coors_pkl_list_all:
                if font_name == os.path.basename(coors_pkl).split('.')[0]:
                    self.can_be_used_font.append(font_name)

        self.font_data = []
        self.same_style_img_dict = {}
        for i, font_name in enumerate(self.can_be_used_font):
            font_pic_pkl = os.path.join(self.pic_path, font_name + self.suffix)
            font_coors_pkl = os.path.join(self.coordinate_path, font_name + self.suffix)
            try:
                font_pics_list = pickle.load(open(font_pic_pkl, 'rb'))
            except Exception as e:
                # print(f"Error loading pic pickle file: {e}")
                # print(font_pic_pkl)
                break
            try:
                font_coors_list = pickle.load(open(font_coors_pkl, 'rb'))
            except Exception as e:
                # print(f"Error loading coors pickle file: {e}")
                # print(font_coors_pkl)
                break
            # 获取同一种字体的self.num_img个图片
            x = self.num_img if len(font_pics_list) >= self.num_img else len(font_pics_list)
            random_index = random.sample(range(len(font_pics_list)), x)
            img_list = []
            for idx in random_index:
                tmp_img = font_pics_list[idx]['img']
                tmp_img = tmp_img / 255.
                img_list.append(tmp_img)
            self.same_style_img_dict[font_name] = img_list

            for pic in font_pics_list:
                char = pic['label']
                """
                文字笔画过多不要了,最多self.max_length,某个3个笔画的文字
                [
                    [(429.0, -43.0, 1, 0)
                     (404.0, -53.0, 0, 0)
                     (549.0, 56.0, 0, 1)],
                    [(967.0, 744.0, 1, 0)
                     (831.0, 774.0, 0, 1)],
                    [(704.0, 130.0, 1, 0)
                     (760.0, 692.0, 0, 1)]
                ]
                单个笔画坐标点太多了也不要了
                """
                try:
                    max_per_stroke_point = max(len(sublist) for sublist in font_coors_list[char])
                    if (char in font_coors_list and len(font_coors_list[char]) <= self.max_stroke and
                            max_per_stroke_point <= self.max_per_stroke_point):
                        self.font_data.append(
                            (i, font_name, pic['label'], pic['img'], font_coors_list[char])
                        )
                        # print('文字:', pic['label'],
                        #       '笔画数量:', str(len(font_coors_list[char])),
                        #       '一个笔画最多坐标点数量:', str(max_per_stroke_point))
                except Exception as e:
                    # 主要是一些中文符号某些字体没有
                    # print(font_name, char)
                    # print(e)
                    pass
        train_size = int(len(self.font_data) * train_percent)
        random.shuffle(self.font_data)
        if is_train:
            self.font_data = self.font_data[:train_size]
        else:
            self.font_data = self.font_data[train_size:]

        self.num_sample = len(self.font_data)

    def __getitem__(self, idx):
        font_nums, font_name, label, char_img, coors = self.font_data[idx]
        try:
            label_id = self.character_std.index(label)
        except Exception as e:
            print(label)
            print(self.character_std)
            print(e)
        char_img = char_img / 255.0
        # 添加通道维度 1 * 64 * 64
        char_img_tensor = torch.tensor(char_img, dtype=torch.float32).unsqueeze(0)

        std_img_tensor = None
        for img in self.img_std:
            if img['label'] == label:
                std_img = img['img'] / 255.0
                std_img_tensor = torch.tensor(std_img, dtype=torch.float32).unsqueeze(0)
        if std_img_tensor is None:
            # 如果没有找到匹配的 label，则使用一个默认的 Tensor
            std_img_tensor = torch.zeros_like(char_img_tensor)

        # 对coors进行padding 使其长度一致 20 * 200 * 4
        # 1.每个字符最多包含的笔画数
        # 2.每个笔画最多包含的点数
        padded_coors = torch.zeros((self.max_stroke, self.max_per_stroke_point, 4), dtype=torch.float32)
        try:
            for i, stroke in enumerate(coors):
                if i >= self.max_stroke:
                    break
                for j, point in enumerate(stroke):
                    if j >= self.max_per_stroke_point:
                        break
                    padded_coors[i, j] = torch.tensor(point, dtype=torch.float32)
            # print(padded_coors)
            # print(padded_coors.shape)
        except Exception as e:
            print(font_name, label)
            print(coors)
            print(e)

        std_coors_tensor = torch.zeros((self.max_stroke, self.max_per_stroke_point, 4), dtype=torch.float32)
        for i, stroke in enumerate(self.coors_std[label]):
            if i >= self.max_stroke:
                break
            for j, point in enumerate(stroke):
                if j >= self.max_per_stroke_point:
                    break
                std_coors_tensor[i, j] = torch.tensor(point, dtype=torch.float32)
        # print(std_coors_tensor)
        # print(std_coors_tensor.shape)

        char_img_list = self.same_style_img_dict[font_name]
        img_list = np.expand_dims(np.array(char_img_list), 1)

        output = {
            'label_id': torch.tensor(label_id, dtype=torch.long),
            'char_img': char_img_tensor,  # torch.Size([1, 64, 64])
            'coordinates': padded_coors,  # torch.Size([20, 200, 4])
            'std_img': std_img_tensor,  # torch.Size([1, 64, 64])
            'std_coors': std_coors_tensor,  # torch.Size([20, 200, 4])
            'same_style_img_list': torch.Tensor(img_list),  # torch.Size([img_num, C, 64, 64])
        }
        return output

    def __len__(self):
        return self.num_sample

    def collect_function(self, batch_data):
        batch_char_img = torch.stack([item['char_img'] for item in batch_data])  # torch.Size([bs, 1, 64, 64])
        batch_coordinates = torch.stack([item['coordinates'] for item in batch_data])  # torch.Size([bs, 20, 200, 4])
        batch_std_img = torch.stack([item['std_img'] for item in batch_data])  # torch.Size([bs, 1, 64, 64])
        batch_std_coors = torch.stack([item['std_coors'] for item in batch_data])  # torch.Size([bs, 20, 200, 4])
        batch_label_ids = torch.tensor([item['label_id'] for item in batch_data], dtype=torch.long)  # torch.Size([bs])
        batch_same_style_img_list = torch.stack(
            [item['same_style_img_list'] for item in batch_data])  # torch.Size([bs, num_img, C, 64, 64])

        return {
            'char_img': batch_char_img,
            'coordinates': batch_coordinates,
            'std_img': batch_std_img,
            'std_coors': batch_std_coors,
            'label_ids': batch_label_ids,
            'same_style_img_list': batch_same_style_img_list,
        }


if __name__ == '__main__':
    fontDataset = FontDataset(is_train=False, is_dev=False)
