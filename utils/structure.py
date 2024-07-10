# pip install easy-media-utils
from tree_utils.struct_tree_out import print_tree

path = r'../../SDT'
path2 = '../z_new_start'
exclude_dirs_set = {'using_files', '__init__.py', 'static', 'LICENSE', 'Generated', 'data', 'test', 'font', 'from',
                    'suit_pics', 'suit_pics2', 'suit_pics3', 'paper', }
print_tree(directory=path, exclude_dirs=exclude_dirs_set)
