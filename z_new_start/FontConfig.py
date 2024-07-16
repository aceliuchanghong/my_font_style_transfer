new_start_config = {
    'test': {
        'z_coordinate_pkl_path': r'D:\aProject\py\my_font_style_transfer\z_new_start\ABtest\files\AB_coors',
        'z_pic_pkl_path': r'D:\aProject\py\my_font_style_transfer\z_new_start\ABtest\files\AB_pics_pkl',
        'content_pkl_path': r'D:\aProject\py\my_font_style_transfer\z_new_start\ABtest\files\new_chinese_content.pkl',
        'character_pkl_path': r'D:\aProject\py\my_font_style_transfer\z_new_start\ABtest\files\new_character_dict.pkl',
        'coors_pkl_path': r'D:\aProject\py\my_font_style_transfer\z_new_start\ABtest\files\LXGWWenKaiGB-Light.pkl',
        'save_model_dir': r'D:\aProject\py\my_font_style_transfer\z_new_start\save_model',
        'PER_BATCH': 8,
        'NUM_THREADS': 0,
        'style_img_path': r'D:\aProject\py\my_font_style_transfer\style_samples',
        'save_pkl_file_path': r'D:\aProject\py\my_font_style_transfer\Saved',
        'suffix': '.png',
    },
    'dev': {
        'z_coordinate_pkl_path': '/mnt/data/llch/files/font_coor_test',
        'z_pic_pkl_path': '/mnt/data/llch/files/font_pics_pkl_test',
        'content_pkl_path': '/mnt/data/llch/files/new_chinese_content.pkl',
        'character_pkl_path': '/mnt/data/llch/files/new_character_dict.pkl',
        'coors_pkl_path': '/mnt/data/llch/files/LXGWWenKaiGB-Light.pkl',
        'save_model_dir': '/mnt/data/llch/my_font_style_transfer/z_new_start/save_model',
        'PER_BATCH': 32,
        'NUM_THREADS': 8,
        'style_img_path': '/mnt/data/llch/my_font_style_transfer/style_samples',
        'save_pkl_file_path': '/mnt/data/llch/Saved',
        'suffix': '.png',
    },
    'train': {
        'seed': 2024,
        'num_epochs': 20,
        'LEARNING_RATE': 1e-4,
        'MAX_STEPS': 200000,
        'SNAPSHOT_BEGIN': 20000,
        'SNAPSHOT_EPOCH': 2000,
        'VALIDATE_BEGIN': 20000,
        'VALIDATE_EPOCH': 2000,
        'IMG_H': 64,
        'IMG_W': 64,
        'd_model': 512,
        'num_head': 8,
        'num_encoder_layers': 2,
        'num_glyph_encoder_layers': 2,
        'num_gly_decoder_layers': 2,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_stroke': 20,
        'max_per_stroke_point': 200,
        'style_img_num': 12,
    },
}
