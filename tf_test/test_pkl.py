import os
import pickle
import matplotlib.pyplot as plt
import lmdb

script = {"CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl'],
          'JAPANESE': ['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
          "ENGLISH": ['CASIA_ENGLISH', 'English_content.pkl']
          }

root = '../data'
dataset = 'CHINESE'
show_num_img = 1
pkl_file = 'C031-f.pkl'

if __name__ == '__main__':
    """
    pkl 文件结构
    item['img'],item['label']
    """
    data_path = os.path.join(root, script[dataset][0])
    content = pickle.load(open(os.path.join(data_path, script[dataset][1]), 'rb'))
    for _ in content:
        if _ in ('一', '吖', '哎', '艾', '鞍'):
            print(_, content[_])
            plt.imshow(content[_], cmap='gray')
            plt.show()
    char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
    print(char_dict)
    all_writer = pickle.load(open(os.path.join(data_path, 'writer_dict.pkl'), 'rb'))
    print(all_writer)
    lmdb_path = os.path.join(data_path, 'test')
    img_path = os.path.join(data_path, 'test_style_samples')
    lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
    print(lmdb.begin(write=False).get('num_sample'.encode('utf-8')).decode())

    test_style_samples01 = pickle.load(open(os.path.join(data_path, 'test_style_samples', pkl_file), 'rb'))
    print(len(test_style_samples01))
    i = 0
    for item in test_style_samples01:
        # print(item,item['img'],item['label'])
        """or
        cv2.imshow("aa", item['img'])
        cv2.waitKey(0)
        """
        print(item)
        plt.imshow(item['img'], cmap='gray')
        plt.show()
        i += 1
        if i >= show_num_img:
            break
