import os
import torch
import wandb
from collections import deque


class ModelCheckpoint:
    def __init__(self, monitor='val_loss', mode='min', save_dir='./checkpoints', save_freq=1, num_checkpoints=2):
        self.monitor = monitor
        self.mode = mode
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.num_checkpoints = num_checkpoints
        self.checkpoints = deque(maxlen=num_checkpoints)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __call__(self, model, epoch, metric_value):
        if epoch % self.save_freq == 0:
            current_checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': metric_value}
            current_metric = current_checkpoint[self.monitor]

            if len(self.checkpoints) == 0 or self._is_better(current_metric, self.checkpoints[0][self.monitor]):
                self._save_checkpoint(current_checkpoint)
                self._write_artifact(current_checkpoint)

    def _is_better(self, current_metric, best_metric):
        if self.mode == 'min':
            return current_metric < best_metric
        elif self.mode == 'max':
            return current_metric > best_metric

    def _save_checkpoint(self, checkpoint):
        epoch = checkpoint['epoch']
        torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch{epoch}.pth'))
        self.checkpoints.append(checkpoint)

    def _write_artifact(self, checkpoint):
        epoch = checkpoint['epoch']
        artifact = wandb.Artifact(f'checkpoint_epoch{epoch}.pth', type='model',
                                  metadata={'metric': checkpoint[self.monitor]})
        artifact.add_file(os.path.join(self.save_dir, f'checkpoint_epoch{epoch}.pth'))
        wandb.run.log_artifact(artifact)
