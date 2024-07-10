from torch.utils.data import DataLoader

from data_loader.loader import ScriptDataset

test_dataset = ScriptDataset(
    '../data', 'CHINESE', False, 15
)
train_loader = DataLoader(test_dataset,
                          batch_size=8,
                          shuffle=True,
                          drop_last=False,
                          collate_fn=test_dataset.collate_fn_,
                          num_workers=0)
