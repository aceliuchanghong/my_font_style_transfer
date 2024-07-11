import torch
import pickle
import os
from torch.utils.data import DataLoader
from z_new_start.FontConfig import new_start_config
from z_new_start.FontModel import FontModel  # 确保导入正确的模型类
from z_new_start.z_utils import restore_coordinates


def load_model(model_path, device, config):
    model = FontModel(
        d_model=config['d_model'],
        num_head=config['num_head'],
        num_encoder_layers=config['num_encoder_layers'],
        num_glyph_encoder_layers=config['num_glyph_encoder_layers'],
        num_gly_decoder_layers=config['num_gly_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        activation="relu",
        normalize_before=True,
        return_intermediate_dec=True,
    )
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_input(image_path, coordinate_path, config):
    with open(image_path, 'rb') as f:
        image_data = pickle.load(f)
    with open(coordinate_path, 'rb') as f:
        coordinate_data = pickle.load(f)

    char_img = image_data['img'] / 255.0  # 正则化
    char_img_tensor = torch.tensor(char_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度

    padded_coors = torch.zeros((config['max_stroke'], config['max_per_stroke_point'], 4), dtype=torch.float32)
    for i, stroke in enumerate(coordinate_data['coordinates']):
        if i >= config['max_stroke']:
            break
        for j, point in enumerate(stroke):
            if j >= config['max_per_stroke_point']:
                break
            padded_coors[i, j] = torch.tensor(point, dtype=torch.float32)
    padded_coors_tensor = padded_coors.unsqueeze(0)  # 添加批次维度

    same_style_img_list = [image_data['img'] / 255.0 for _ in range(config['num_img'])]
    same_style_img_tensor = torch.tensor(same_style_img_list, dtype=torch.float32).unsqueeze(1).unsqueeze(
        0)  # 添加批次和通道维度

    return char_img_tensor, padded_coors_tensor, same_style_img_tensor


def generate_coordinates(model, char_img_tensor, padded_coors_tensor, same_style_img_tensor, device, config):
    char_img_tensor = char_img_tensor.to(device)
    padded_coors_tensor = padded_coors_tensor.to(device)
    same_style_img_tensor = same_style_img_tensor.to(device)

    with torch.no_grad():
        pred_sequence = model(same_style_img_tensor, padded_coors_tensor, char_img_tensor)

    pred_coordinates = pred_sequence.cpu().numpy().squeeze()

    restored_coordinates = restore_coordinates(torch.tensor(pred_coordinates), config['max_stroke'],
                                               config['max_per_stroke_point'])
    return restored_coordinates


if __name__ == '__main__':
    # 配置和路径
    conf = new_start_config
    data_conf = conf['dev'] if os.getenv('ENV') == 'dev' else conf['test']
    model_path = 'path/to/pretrained_model.pt'
    image_path = 'path/to/image_data.pkl'
    coordinate_path = 'path/to/coordinate_data.pkl'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    model = load_model(model_path, device, conf['train'])

    # 预处理输入数据
    char_img_tensor, padded_coors_tensor, same_style_img_tensor = preprocess_input(image_path, coordinate_path,
                                                                                   conf['train'])

    # 生成坐标
    generated_coordinates = generate_coordinates(model, char_img_tensor, padded_coors_tensor, same_style_img_tensor,
                                                 device, conf['train'])

    print(generated_coordinates)
