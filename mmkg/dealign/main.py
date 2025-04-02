from collections import defaultdict

import torch
from torch import optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from mmkg.dealign.DESAlign import DESAlign


class Runner:
    def __init__(self, writer=None, logger=None):
        self.writer = writer
        self.logger = logger
        self.scaler = GradScaler()  # 自动梯度缩放
        self.model_choice()
        if self.args.only_test:
            self.dataloader_init(test_set=self.test_set)
        else:
            self.dataloader_init(train_set=self.train_set, eval_set=self.eval_set, test_set=self.test_set)
            self.model_list = [self.model]

    def model_choice(self, load_name=None):
        self.model = DESAlign(self.kgs, self.args)

    def optim_init(self, opt, total_epoch=None, accumulation_step=None):
        step_per_epoch = len(self.train_dataloader)
        if total_epoch is not None:
            opt.total_steps = int(step_per_epoch * total_epoch)
        opt.warmup_steps = int(step_per_epoch * 0.15)
        self.logger.info(f"total_steps: {opt.total_steps}")
        self.logger.info(f"warmup_steps: {opt.warmup_steps}")
        self.logger.info(f"weight_decay: {opt.weight_decay}")
        self.optimizer = optim.AdamW(opt.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=int(opt.warmup_steps / accumulation_step),
                                                         num_training_steps=int(opt.total_steps / accumulation_step))

    def run(self):
        self.epoch = -1
        self.step = 1
        self.curr_loss = 0
        self.curr_loss_dic = defaultdict(float)
        with tqdm(total=self.args.epoch) as bar:
            for i in range(self.args.epoch):
                self.epoch += 1
                torch.cuda.empty_cache()
                self.train(bar)

    def train(self, tqdm):
        self.model.train()
        accumulation_steps = self.args.accumulation_steps  # 梯度积累的步数
        curr_loss = 0
        for batch in self.train_dataloader:
            loss, output = self.model(batch)
            loss = loss / accumulation_steps  # 平均损失, 适配梯度累积
            self.scaler.scale(loss).backward()  # 反向传播(梯度缩放, 避免数值问题)
            self.step += 1
            curr_loss += loss.item()
            self.output_statistic(loss, output)

            if self.step % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                for model in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.clip)

                scale = self.scaler.get_scale()  # 获取缩放因子
                self.scaler.step(self.optimizer)  # 更新优化器参数
                self.scaler.update()  # 更新缩放因子

                skip_lr_schem = (scale > self.scaler.get_scale())
                if not skip_lr_schem:
                    self.scheduler.step()
                self.lr = self.scheduler.get_last_lr()[-1]  # 记录当前学习率
                self.writer.add_scalars("lr", {"lr": self.lr}, self.step)
                for model in self.model_list:
                    model.zero_grad(set_to_none=True)
        return curr_loss

    def output_statistic(self, loss, output):
        self.curr_loss += loss.item()
        if output is None:
            return
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
