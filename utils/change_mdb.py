from utils.judge_font import get_files
from utils.util import writeCache
import os
import lmdb
import pickle

script = {"CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl'],
          'JAPANESE': ['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
          "ENGLISH": ['CASIA_ENGLISH', 'English_content.pkl']
          }

root = '../data'
dataset = 'CHINESE'

if __name__ == '__main__':
    data_path = os.path.join(root, script[dataset][0])
    lmdb_path = os.path.join(data_path, 'train')
    my_pkl_path = r'D:\aProject\py\SDT\utils\suit_pics3'
    new_env = lmdb.open(my_pkl_path, map_size=1073741824)
    lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
    new_num_sample = 0
    new_cache = {}

    num_sample = lmdb.begin(write=False).get('num_sample'.encode('utf-8')).decode()
    print('num_sample:', lmdb.begin(write=False).get('num_sample'.encode('utf-8')).decode())
    filter0 = []
    for x in get_files(my_pkl_path, 'pkl'):
        filter0.append(os.path.basename(x).split('.')[0])

    with lmdb.begin(write=False) as txn:
        for i in range(int(num_sample)):
            data = pickle.loads(txn.get(str(i).encode('utf-8')))
            tag_char, coords, fname = data['tag_char'], data['coordinates'], data['fname'].split('.')[0]
            if fname in filter0:
                new_data = {"tag_char": tag_char, "coordinates": coords, "fname": fname + ".pot"}
                print(fname)
                data_byte = pickle.dumps(data)
                data_id = str(new_num_sample).encode('utf-8')
                new_cache[data_id] = data_byte
                new_num_sample += 1
    new_cache['num_sample'.encode('utf-8')] = str(new_num_sample).encode()
    writeCache(new_env, new_cache)
