import time
import cv2
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel

import wandb
from clrnet.utils.logging import init_wandb, wandb_log_train, wandb_log_val
from clrnet.utils.recorder import SmoothedValue




class CLRNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')
            
            return loss
                
        # Logging on W&B
        wandb_log_train(self.recorder.loss_stats['loss'].avg, self.recorder.lr, self.recorder.epoch)
    
    def train_dataloader(self):
     return build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

    def test_dataloader(self):
     return build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=False)

    def val_dataloader(self):
     return build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=False)

    def training_step(self,batch,batch_idx):
    
            self.recorder.epoch = batch_idx
            loss=self.train_epoch(batch_idx, batch)
            return loss

    def training_epoch_end(self, training_step_outputs,batch_idx):
        all_preds = torch.stack(training_step_outputs)
        if (batch_idx +
                    1) % self.cfg.save_ep == 0 or batch_idx == self.cfg.epochs - 1:
                self.save_ckpt()

    def test_step(self,batch,batch_idx):
        predictions = []
        for i, data in enumerate(tqdm(batch, desc=f'Testing')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                batch.dataset.view(output, data['meta'])

        metric = batch.dataset.evaluate(predictions,
                                                   self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))
        
        return metric

    def validation_step(self,batch,batch_idx):
        predictions = []
        for i, data in enumerate(tqdm(batch, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                batch.dataset.view(output, data['meta'])

        metric = batch.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))
        
        # Logging on W&B
        wandb_log_val(metric, self.recorder.epoch)

        return metric
        

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)
        
        
        ckpt_name = 'best' if is_best else self.recorder.epoch
        wandb.save(os.path.join(os.path.join(self.recorder.work_dir, 'ckpt'), '{}.pth'.format(ckpt_name)))
