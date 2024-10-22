new_start_config = {
    'test': {
        'z_coordinate_pkl_path': r'D:\aProject\py\my_font_style_transfer\z_new_start\ABtest\files\AB_coors',
        'z_pic_pkl_path': r'D:\aProject\py\my_font_style_transfer\z_new_start\ABtest\files\AB_pics_pkl',
        'content_pkl_path': r'D:\aProject\py\my_font_style_transfer\z_new_start\ABtest\files\new_chinese_content.pkl',
        'character_pkl_path': r'D:\aProject\py\my_font_style_transfer\z_new_start\ABtest\files\new_character_dict.pkl',
        'coors_pkl_path': r'D:\soft\FontForgeBuilds\ll\FZHTJW-xx2.pkl',
        'save_model_dir': r'D:\aProject\py\my_font_style_transfer\z_new_start\save_model',
        'PER_BATCH': 2,
        'NUM_THREADS': 0,
        'style_img_path': r'D:\aProject\py\my_font_style_transfer\style_samples',
        'style_pkl_file_path': r'D:\aProject\py\my_font_style_transfer\Saved',
        'suffix': '.png',
    },
    'dev': {
        'z_coordinate_pkl_path': '/mnt/data/llch/files/gen_full/coor/lch_coor',
        'z_pic_pkl_path': '/mnt/data/llch/files/gen_full/pics/lch_pics_pkl',
        'content_pkl_path': '/mnt/data/llch/files/gen_full/new_chinese_content.pkl',
        'character_pkl_path': '/mnt/data/llch/files/gen_full/new_character_dict.pkl',
        'coors_pkl_path': '/mnt/data/llch/files/gen_full/new_std_coor.pkl',
        'save_model_dir': '/mnt/data/llch/my_font_style_transfer/z_new_start/save_model',
        'PER_BATCH': 24,
        'NUM_THREADS': 8,
        'style_img_path': '/mnt/data/llch/my_font_style_transfer/style_samples',
        'style_pkl_file_path': '/mnt/data/llch/my_font_style_transfer/Saved',
        'suffix': '.png',
    },
    'train': {
        'seed': 2024,
        'num_epochs': 20,
        'LEARNING_RATE': 1e-4,
        'MAX_STEPS': 13000,
        'SNAPSHOT_BEGIN': 1000,
        'SNAPSHOT_EPOCH': 499,
        'd_model': 512,
        'num_head': 8,
        'num_encoder_layers': 2,
        'num_glyph_encoder_layers': 2,
        'num_gly_decoder_layers': 2,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_stroke': 20,
        'max_per_stroke_point': 200,
        'style_img_num': 24,
        'keys': 'dropout'
    },
}
