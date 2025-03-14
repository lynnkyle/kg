class Trainer(object):
    def __init__(self, args=None, logger=None, data_loader=None,
                 model=None, train_times=1000, alpha=0.5, use_gpu=True,
                 opt_method='sgd', save_steps=None, checkpoint_dir=None,
                 train_mode='adp', beta=0.5):
        self.data_loader = data_loader
        self.model = model
        self.use_gpu = use_gpu

    def run(self):
        if self.use_gpu:
            self.model.cuda()
