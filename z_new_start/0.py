from z_new_start.FontDataset import FontDataset
from z_new_start.FontUtils import CoorsRender, _get_coors_decode
import torch

tensor0 = torch.randn(1, 20, 200, 4)
tensor1 = torch.randn(1, 20, 200, 4)
pred = []
pred.append(tensor0)
pred.append(tensor1)
images = torch.randn(1, 12, 1, 64, 64)
gd = FontDataset(is_train=False, is_dev=False)
outputs = _get_coors_decode(CoorsRender(), p=pred, images=images, dropout=0.1, gd=gd)
print(outputs)
