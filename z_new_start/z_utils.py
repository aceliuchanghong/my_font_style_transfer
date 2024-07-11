import torch


def restore_coordinates(padded_coors, max_stroke=20, max_per_stroke_point=200):
    """
    将填充后的坐标张量还原为原始的字符坐标格式。
    Args:
        padded_coors (torch.Tensor): 填充后的坐标张量，形状为 [max_stroke, max_per_stroke_point, 4]
        max_stroke (int): 最大笔画数
        max_per_stroke_point (int): 每个笔画的最大点数
    Returns:
        list: 还原后的字符坐标格式
    """
    restored_coordinates = []

    for i in range(max_stroke):
        stroke = []
        for j in range(max_per_stroke_point):
            point = padded_coors[i, j]
            if torch.all(point == 0):
                break  # 如果当前点全为0，则跳过该点
            stroke.append((point[0].item(), point[1].item(), int(point[2].item()), int(point[3].item())))
        if stroke:
            restored_coordinates.append(stroke)

    return restored_coordinates


if __name__ == '__main__':
    # 测试示例
    padded_coors = torch.tensor([
        [[429.0, -43.0, 1.0, 0.0], [404.0, -53.0, 0.0, 0.0], [549.0, 56.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        [[967.0, 744.0, 1.0, 0.0], [831.0, 774.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[704.0, 130.0, 1.0, 0.0], [760.0, 692.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ])

    restored_coordinates = restore_coordinates(padded_coors, 3, 4)
    print(restored_coordinates)
