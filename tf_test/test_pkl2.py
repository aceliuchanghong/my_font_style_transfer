import matplotlib.pyplot as plt
import pickle


def read_pkl(file_path, show_pic_num=0):
    test_style_samples01 = pickle.load(open(file_path, 'rb'))
    print("total pics:", len(test_style_samples01))
    i = 0
    for item in test_style_samples01:
        plt.imshow(item['img'], cmap='gray')
        plt.show()
        i += 1
        if i >= show_pic_num:
            print(item)
            break


if __name__ == '__main__':
    file_path = r'D:\aProject\py\SDT\utils\suit_pics3\001.pkl'
    read_pkl(file_path, 1)
