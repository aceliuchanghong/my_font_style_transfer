from data_loader.loader import UserDataset
import torch
from torch.utils.data import DataLoader

test_dataset = UserDataset(
    root='../data', dataset='CHINESE', style_path='../style_samples'
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True,
    sampler=None,
    drop_last=False,
    num_workers=0
)
data_iter = iter(test_loader)
batch_samples = len(test_loader)
for batch in range(batch_samples):
    print('batch:', batch)
    data = next(data_iter)
    print(data)
    break
