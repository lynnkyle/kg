import torch

path_1 = '/home/ps/lzy/kg/mmkg/mygo/tokens/visual.pth'
path_2 = '/home/ps/lzy/kg/mmkg/mygo/tokens/visual_vqgan.pth'
model = torch.load(path_2)
print(model.shape)
