import random
from utils.util import normalize_xys
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
import lmdb
from utils.util import corrds2xys
import codecs
import glob
import cv2

"""
数据预处理
transforms.ToTensor(): 
这个函数将PIL图像或NumPy的ndarray转换为形状为[C, H, W]的PyTorch张量，其中C是通道数，H是图像高度，W是图像宽度
Normalize
每个通道进行归一化处理
这个例子中，归一化是通过减去0.5然后除以0.5来完成的，这是一个常见的归一化方法，它将数据缩放到[-1, 1]的范围内。

正确的做法是先计算训练数据的均值和标准差，然后使用这些值来初始化transforms.Normalize。例如：
# 假设train_data是一个包含训练图像的数据集
train_data = datasets.ImageFolder(root='path_to_train_data', transform=transforms.ToTensor())

# 计算均值和标准差
mean = torch.mean(train_data.data / 255.0)
std = torch.std(train_data.data / 255.0)

# 使用计算得到的均值和标准差初始化Normalize
transform_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(mean,), std=(std,))
])
"""
transform_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))
])

script = {"CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl'],
          'JAPANESE': ['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
          "ENGLISH": ['CASIA_ENGLISH', 'English_content.pkl'],
          # lch add
          "CHINESE_TEST": ['CASIA_CHINESE_TEST', 'Chinese_content.pkl'],
          }


class ScriptDataset(Dataset):
    def __init__(self, root='data', dataset='CHINESE', is_train=True, num_img=15):
        data_path = os.path.join(root, script[dataset][0])
        self.dataset = dataset
        self.content = pickle.load(open(os.path.join(data_path, script[dataset][1]), 'rb'))  # content samples
        self.char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
        self.all_writer = pickle.load(open(os.path.join(data_path, 'writer_dict.pkl'), 'rb'))
        self.is_train = is_train
        if self.is_train:
            lmdb_path = os.path.join(data_path, 'train')  # online characters
            self.img_path = os.path.join(data_path, 'train_style_samples')  # style samples
            self.num_img = num_img * 2
            self.writer_dict = self.all_writer['train_writer']
        else:
            lmdb_path = os.path.join(data_path, 'test')  # online characters
            self.img_path = os.path.join(data_path, 'test_style_samples')  # style samples
            self.num_img = num_img
            self.writer_dict = self.all_writer['test_writer']
        if not os.path.exists(lmdb_path):
            raise IOError("input the correct lmdb path:", lmdb_path)

        self.lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
        # max_len 变量用于控制在创建数据集时，是否过滤掉具有更多轨迹点的字符
        if script[dataset][0] in ("CASIA_CHINESE", "CASIA_CHINESE_TEST"):
            self.max_len = -1  # Do not filter characters with many trajectory points
        else:  # Japanese, Indic, English
            self.max_len = 150

        self.all_path = {}
        for pkl in os.listdir(self.img_path):
            writer = pkl.split('.')[0]
            self.all_path[writer] = os.path.join(self.img_path, pkl)

        with self.lmdb.begin(write=False) as txn:
            self.num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())
            if self.max_len <= 0:
                self.indexes = list(range(0, self.num_sample))
            else:
                print('Filter the characters containing more than max_len points')
                self.indexes = []
                for i in range(self.num_sample):
                    data_id = str(i).encode('utf-8')
                    data_byte = txn.get(data_id)
                    coords = pickle.loads(data_byte)['coordinates']
                    if len(coords) < self.max_len:
                        self.indexes.append(i)
                    else:
                        pass

    def __getitem__(self, index):
        index = self.indexes[index]
        with self.lmdb.begin(write=False) as txn:
            data = pickle.loads(txn.get(str(index).encode('utf-8')))
            tag_char, coords, fname = data['tag_char'], data['coordinates'], data['fname']
        char_img = self.content[tag_char]  # content samples
        char_img = char_img / 255.  # Normalize pixel values between 0.0 and 1.0
        writer = data['fname'].split('.')[0]
        img_path_list = self.all_path[writer]
        with open(img_path_list, 'rb') as f:
            style_samples = pickle.load(f)
        img_list = []
        img_label = []
        random_indexs = random.sample(range(len(style_samples)), self.num_img)
        for idx in random_indexs:
            tmp_img = style_samples[idx]['img']
            tmp_img = tmp_img / 255.
            tmp_label = style_samples[idx]['label']
            img_list.append(tmp_img)
            if self.dataset == 'JAPANESE':
                tmp_label = bytes.fromhex(tmp_label[5:])
                tmp_label = codecs.decode(tmp_label, "cp932")
            img_label.append(tmp_label)
        img_list = np.expand_dims(np.array(img_list), 1)  # [N, C, H, W], C=1
        coords = normalize_xys(coords)  # Coordinate Normalization

        # Convert absolute coordinate values into relative ones
        coords[1:, :2] = coords[1:, :2] - coords[:-1, :2]

        writer_id = self.writer_dict[fname]
        character_id = self.char_dict.find(tag_char)
        label_id = []
        for i in range(self.num_img):
            label_id.append(self.char_dict.find(img_label[i]))
        return {'coords': torch.Tensor(coords),
                'character_id': torch.Tensor([character_id]),
                'writer_id': torch.Tensor([writer_id]),
                'img_list': torch.Tensor(img_list),
                'char_img': torch.Tensor(char_img),
                'img_label': torch.Tensor([label_id])}

    def __len__(self):
        return len(self.indexes)

    def collate_fn_(self, batch_data):
        bs = len(batch_data)
        # 找到 batch 中最长的序列长度，并加1（因为需要在末尾填充一个结束状态）
        max_len = max([s['coords'].shape[0] for s in batch_data]) + 1
        output = {'coords': torch.zeros((bs, max_len, 5)),  # (batch_size, max_len, 5)的张量，表示每个样本的坐标和状态
                  # (x, y, state_1, state_2, state_3)==> (x,y,pen_down,pen_up,pen_end) 下笔、提笔、终止
                  'coords_len': torch.zeros((bs,)),  # 每个样本的实际长度
                  'character_id': torch.zeros((bs,)),
                  'writer_id': torch.zeros((bs,)),
                  'img_list': [],
                  'char_img': [],
                  'img_label': []}
        output['coords'][:, :, -1] = 1  # 用笔的结束状态填充

        for i in range(bs):
            s = batch_data[i]['coords'].shape[0]
            output['coords'][i, :s] = batch_data[i]['coords'] # 填充当前样本的坐标和状态
            output['coords'][i, 0, :2] = 0  # 在第一个token处放置下笔状态
            output['coords_len'][i] = s
            output['character_id'][i] = batch_data[i]['character_id']
            output['writer_id'][i] = batch_data[i]['writer_id']
            output['img_list'].append(batch_data[i]['img_list'])
            output['char_img'].append(batch_data[i]['char_img'])
            output['img_label'].append(batch_data[i]['img_label'])
        output['img_list'] = torch.stack(output['img_list'], 0)  # -> (B, num_img, 1, H, W)
        temp = torch.stack(output['char_img'], 0)
        output['char_img'] = temp.unsqueeze(1)
        output['img_label'] = torch.cat(output['img_label'], 0)
        output['img_label'] = output['img_label'].view(-1, 1).squeeze()
        return output


"""
 loading generated online characters for evaluating the generation quality
"""


class Online_Dataset(Dataset):
    def __init__(self, data_path):
        lmdb_path = os.path.join(data_path, 'test')
        print("loading characters from", lmdb_path)
        if not os.path.exists(lmdb_path):
            raise IOError("input the correct lmdb path")

        self.char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
        self.writer_dict = pickle.load(open(os.path.join(data_path, 'writer_dict.pkl'), 'rb'))
        self.lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)

        with self.lmdb.begin(write=False) as txn:
            self.num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())
            self.indexes = list(range(0, self.num_sample))

    def __getitem__(self, index):
        with self.lmdb.begin(write=False) as txn:
            data = pickle.loads(txn.get(str(index).encode('utf-8')))
            character_id, coords, writer_id, coords_gt = data['character_id'], \
                data['coordinates'], data['writer_id'], data['coords_gt']
        try:
            coords, coords_gt = corrds2xys(coords), corrds2xys(coords_gt)
        except:
            print('Error in character format conversion')
            return self[index + 1]
        return {'coords': torch.Tensor(coords),
                'character_id': torch.Tensor([character_id]),
                'writer_id': torch.Tensor([writer_id]),
                'coords_gt': torch.Tensor(coords_gt)}

    def __len__(self):
        return len(self.indexes)

    def collate_fn_(self, batch_data):
        bs = len(batch_data)
        max_len = max([s['coords'].shape[0] for s in batch_data])
        max_len_gt = max([h['coords_gt'].shape[0] for h in batch_data])
        output = {'coords': torch.zeros((bs, max_len, 5)),  # preds -> (x,y,state) 
                  'coords_gt': torch.zeros((bs, max_len_gt, 5)),  # gt -> (x,y,state)
                  'coords_len': torch.zeros((bs,)),
                  'len_gt': torch.zeros((bs,)),
                  'character_id': torch.zeros((bs,)),
                  'writer_id': torch.zeros((bs,))}

        for i in range(bs):
            s = batch_data[i]['coords'].shape[0]
            output['coords'][i, :s] = batch_data[i]['coords']
            h = batch_data[i]['coords_gt'].shape[0]
            output['coords_gt'][i, :h] = batch_data[i]['coords_gt']
            output['coords_len'][i], output['len_gt'][i] = s, h
            output['character_id'][i] = batch_data[i]['character_id']
            output['writer_id'][i] = batch_data[i]['writer_id']
        return output


class UserDataset(Dataset):
    def __init__(self, root='data', dataset='CHINESE', style_path='style_samples'):
        data_path = os.path.join(root, script[dataset][0])
        self.content = pickle.load(open(os.path.join(data_path, script[dataset][1]), 'rb'))  # content samples
        self.char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
        self.style_path = glob.glob(style_path + '/*.[jp][pn]g')

    def __len__(self):
        return len(self.char_dict)

    def __getitem__(self, index):
        char = self.char_dict[index]  # content samples
        char_img = self.content[char]
        char_img = char_img / 255.  # Normalize pixel values between 0.0 and 1.0
        img_list = []
        # print(self.style_path)
        for idx in range(len(self.style_path)):
            style_img = cv2.imdecode(np.fromfile(self.style_path[idx], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            # style_img = cv2.imread(self.style_path[idx], flags=0)
            style_img = cv2.resize(style_img, (64, 64))
            style_img = style_img / 255.
            img_list.append(style_img)
        img_list = np.expand_dims(np.array(img_list), 1)

        return {'char_img': torch.Tensor(char_img).unsqueeze(0),
                'img_list': torch.Tensor(img_list),
                'char': char}
