import torch
import os
import time
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FontTrainer:
    def __init__(self,
                 model, train_loader, valid_loader, criterion, optimizer, device, train_conf, data_conf):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_conf = train_conf
        self.data_conf = data_conf
        self.best_loss = float('inf')
        self.scaler = torch.cuda.amp.GradScaler()
        self.accumulation_steps = 16  # 累积梯度的步数

    def train(self):
        num_epochs = self.train_conf['num_epochs']
        max_steps = self.train_conf['MAX_STEPS']
        logger.info(f"Start training epochs: {num_epochs}")
        start_time = time.time()
        step = 0
        for epoch in range(num_epochs):
            train_loader_iter = iter(self.train_loader)
            try:
                pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
                while True:
                    if max_steps and step >= max_steps:
                        logger.info(
                            f"Reached max steps: {max_steps}. Stopping training. The epoch:{epoch}. Total time: {time.time() - start_time:.2f}s")
                        return
                    data = next(train_loader_iter)
                    self._train_iter(data, step)
                    self._save_checkpoint(step)
                    val_loss = self._valid_iter(step)
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self._save_best_model(step, val_loss)
                    self._progress(step, num_epochs, start_time)
                    step += 1
                    pbar.update(1)
            except StopIteration:
                pbar.close()
            except Exception as e:
                logger.error(f"Error: {e}\ntrain_loader_iter_epoch failed:{epoch}")
                return
        logger.info(f"Training finished. Total time: {time.time() - start_time:.2f}s")

    def _train_iter(self, data, step):
        self.model.train()
        iter_time = time.time()
        # 仅在需要时将数据放入 GPU
        char_img_gt = data['char_img'].to(self.device, non_blocking=True)
        coordinates_gt = data['coordinates'].to(self.device, non_blocking=True)
        std_coors = data['std_coors'].to(self.device, non_blocking=True)
        same_style_img_list = data['same_style_img_list'].to(self.device, non_blocking=True)
        # 先不放不需要的
        # std_img = data['std_img'].to(self.device)
        # label_ids = data['label_ids'].to(self.device)

        # PyTorch 提供的自动混合精度训练
        with torch.cuda.amp.autocast():
            # torch.Size([bs, num, c, 64, 64])
            # torch.Size([bs, 20, 200, 4])
            # torch.Size([bs, c, 64, 64])
            predict = self.model(same_style_img_list, std_coors, char_img_gt)
            assert predict.shape == coordinates_gt.shape, f"Shape mismatch: predict {predict.shape}, coordinates_gt {coordinates_gt.shape}"
            loss = self.criterion(predict, coordinates_gt)
            loss = loss / self.accumulation_steps

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # 增加梯度累加
        if (step + 1) % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        logger.info(f"Step {step}, Iteration time: {time.time() - iter_time:.4f}s, Loss: {loss.item():.4f}")

        del data, predict, loss
        torch.cuda.empty_cache()

    def _valid_iter(self, step):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.valid_loader:
                char_img_gt = data['char_img'].to(self.device, non_blocking=True)
                coordinates_gt = data['coordinates'].to(self.device, non_blocking=True)
                std_coors = data['std_coors'].to(self.device, non_blocking=True)
                same_style_img_list = data['same_style_img_list'].to(self.device, non_blocking=True)

                # std_img = data['std_img'].to(self.device)
                # label_ids = data['label_ids'].to(self.device)

                with torch.cuda.amp.autocast():  # 使用自动混合精度
                    predict = self.model(same_style_img_list, std_coors, char_img_gt)
                    loss = self.criterion(predict, coordinates_gt)
                    total_loss += loss.item()
        avg_loss = total_loss / len(self.valid_loader)
        logger.info(f"Validation loss at step {step}: {avg_loss:.4f}")
        return avg_loss

    def _save_checkpoint(self, step):
        if step >= self.train_conf['SNAPSHOT_BEGIN'] and step % self.train_conf['SNAPSHOT_EPOCH'] == 0:
            checkpoint_path = os.path.join(self.data_conf['save_model_dir'], f'checkpoint_step_{step}.pt')
            model_state_dict = self.model.module.state_dict() if isinstance(self.model,
                                                                            torch.nn.DataParallel) else self.model.state_dict()
            torch.save({
                'step': step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at step {step} to {checkpoint_path}")

            # 只保留最近的10个检查点
            checkpoints = sorted(
                [f for f in os.listdir(self.data_conf['save_model_dir']) if f.startswith('checkpoint_step_')])
            for old_checkpoint in checkpoints[:-10]:
                os.remove(os.path.join(self.data_conf['save_model_dir'], old_checkpoint))

    def _save_best_model(self, step, loss):
        best_model_path = os.path.join(self.data_conf['save_model_dir'], 'best_model.pt')
        model_state_dict = self.model.module.state_dict() if isinstance(self.model,
                                                                        torch.nn.DataParallel) else self.model.state_dict()
        torch.save({
            'step': step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, best_model_path)
        logger.info(f"Best model saved at step {step} with validation loss {loss:.4f} to {best_model_path}")

    def _progress(self, step, num_epochs, start_time):
        elapsed_time = time.time() - start_time
        steps_per_epoch = len(self.train_loader)
        total_steps = num_epochs * steps_per_epoch
        eta = (elapsed_time / step) * (total_steps - step) if step > 0 else 0
        eta_minutes, eta_seconds = divmod(eta, 60)
        logger.info(
            f"Step {step}/{total_steps}, ETA: {int(eta_minutes)}m {int(eta_seconds)}s, Elapsed time: {elapsed_time:.2f}s")
