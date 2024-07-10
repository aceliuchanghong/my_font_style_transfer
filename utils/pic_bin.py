from PIL import Image
import os

# 设置二值化阈值
threshold = 128
current_dir = 'font/result/'
save_dir = '../style_samples2/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if __name__ == '__main__':
    for filename in os.listdir(current_dir):
        if filename.endswith(('.jpg', '.png')):
            img = Image.open(current_dir + filename)
            gray_img = img.convert('L')
            binary_img = gray_img.point(lambda x: 255 if x > threshold else 0, '1')
            binary_filename = filename.rsplit('.', 1)[0] + '_binary.' + filename.rsplit('.', 1)[1]
            binary_img.save(save_dir + binary_filename)

            print(f"保存为：{binary_filename}")
