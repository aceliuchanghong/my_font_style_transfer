import os
import lmdb
import pickle

from utils.util import coords_render

script = {"CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl'],
          'JAPANESE': ['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
          "ENGLISH": ['CASIA_ENGLISH', 'English_content.pkl']
          }

root = '../data'
dataset = 'CHINESE'
num_img = 15
index = 3
max_len = 150
if __name__ == '__main__':
    data_path = os.path.join(root, script[dataset][0])
    lmdb_path = os.path.join(data_path, 'train')
    # lmdb_path = r'D:\aProject\py\SDT\utils\suit_pics3'
    print(lmdb_path)
    """
    max_readers=8: 允许最多8个读取器并发访问数据库。
    readonly=True: 以只读方式打开数据库。
    lock=False: 不使用文件锁，这样可以在多个进程中只读访问数据库。
    readahead=False: 关闭预读取功能，以降低对内存的占用。
    meminit=False: 不预先初始化内存。
    LMDB 数据库是由一个环境(environment)和多个事务(transactions)组成的，环境包括数据文件和锁文件。
    从 data.mdb 文件中读取数据，而 lock.mdb 文件是用来进行事务处理的文件锁。当设置 lock=False 时，锁文件不会被使用
    """
    lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
    print(lmdb.begin(write=False).get('num_sample'.encode('utf-8')).decode())
    with lmdb.begin(write=False) as txn:
        num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())
        for i in range(index):
            data = pickle.loads(txn.get(str(i).encode('utf-8')))
            # print(data)
            tag_char, coords, fname = data['tag_char'], data['coordinates'], data['fname']
            print(str(i) + "个:\ntag_char: {}\ncoords_shape: {}\nfname: {}".format(tag_char, coords.shape, fname))
            print("coords:\n", coords)

