from utils.judge_font import get_files
import os
import shutil

dir_list = [
    r'C:\Users\liuch\Pictures\02\Chinese_User_02',
    r'C:\Users\liuch\Pictures\generate_new_01',
    r'C:\Users\liuch\Pictures\generate_new_02',
    r'C:\Users\liuch\Pictures\generate_lch',
    r'C:\Users\liuch\Pictures\07'
]
save_dir = r'C:\Users\liuch\Pictures\show'
name_list = ['点', '燃', '思', '想', '的', '路']

if __name__ == '__main__':
    for type, directory_path in enumerate(dir_list, start=1):
        print("working on:", directory_path)
        print(type, directory_path)
        filelist = get_files(directory_path, '.png')
        index = 0
        for i in name_list:
            for file in filelist:
                if i in os.path.basename(file):
                    new_filename = f"{type}_{index}_{i}.png"
                    shutil.copy(file, os.path.join(save_dir, new_filename))
            index += 1
