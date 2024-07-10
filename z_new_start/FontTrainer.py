import torch
import os
import time
import logging

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

    def train(self):
        num_epochs = self.train_conf['num_epochs']
        max_steps = self.train_conf['MAX_STEPS']
        logger.info(f"Start training epochs: {num_epochs}")

        start_time = time.time()
        step = 0
        for epoch in range(num_epochs):
            train_loader_iter = iter(self.train_loader)
            try:
                while True:
                    if max_steps and step >= max_steps:
                        logger.info(
                            f"Reached max steps: {max_steps}. Stopping training.The epoch:{epoch}.Total time: {time.time() - start_time}")
                        return
                    data = next(train_loader_iter)
                    self._train_iter(data, step)
                    self._save_checkpoint(step)
                    self._valid_iter(step)
                    self._progress(step, num_epochs, start_time)
                    step += 1
            except StopIteration:
                # 所有数据都被消费完，继续下一个 epoch
                pass
            except Exception as e:
                logger.error(f"Error: {e}\ntrain_loader_iter_epoch failed:{epoch}")
                return
        logger.info(f"Training finished. Total time: {time.time() - start_time}")

    def _train_iter(self, data, epoch):
        # 在没有使用数据的情况下调用的，但这是为了确保在每次训练迭代开始时，模型都处于正确的训练模式
        # 执行一次训练迭代，包括前向传播、计算损失、反向传播和更新模型参数。
        self.model.train()

        iter_time = time.time()
        char_img_gt = data['char_img'].to(self.device)  # torch.Size([8, 1, 64, 64])
        coordinates_gt = data['coordinates'].to(self.device)  # torch.Size([8, 20, 200, 4])
        std_img = data['std_img'].to(self.device)  # torch.Size([8, 1, 64, 64])
        std_coors = data['std_coors'].to(self.device)  # torch.Size([8, 20, 200, 4])
        label_ids = data['label_ids'].to(self.device)  # torch.Size([8])

        predict = self.model(char_img_gt, std_coors)
        loss = self.criterion(predict, coordinates_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Epoch {epoch}, Iteration time: {time.time() - iter_time:.4f} seconds, Loss: {loss.item():.4f}")

    def _valid_iter(self, step):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.valid_loader:
                char_img_gt = data['char_img'].to(self.device)
                coordinates_gt = data['coordinates'].to(self.device)
                std_img = data['std_img'].to(self.device)
                std_coors = data['std_coors'].to(self.device)
                label_ids = data['label_ids'].to(self.device)
                predict = self.model(char_img_gt, std_coors)
                loss = self.criterion(predict, coordinates_gt)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.valid_loader)
        logger.info(f"Validation loss at epoch {step}: {avg_loss:.4f}")
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
        logger.info(f"Best model saved at epoch {step} with validation loss {loss} to {best_model_path}")

    def _progress(self, step, num_epochs, start_time):
        elapsed_time = time.time() - start_time
        steps_per_epoch = len(self.train_loader)
        total_steps = num_epochs * steps_per_epoch
        eta = (elapsed_time / step) * (total_steps - step) if step > 0 else 0
        eta_minutes, eta_seconds = divmod(eta, 60)
        logger.info(
            f"Step {step}/{total_steps}, ETA: {int(eta_minutes)} minutes {int(eta_seconds)} seconds, Elapsed time: {elapsed_time:.2f} seconds")
