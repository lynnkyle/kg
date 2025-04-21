from easydict import EasyDict

config = EasyDict()
config.batch_size = 32
config.learning_rate = 0.01
config.model = EasyDict()
config.model.name = 'ResNet50'
config.model.depth = 50
print(config.batch_size)
print(config.learning_rate)
print(config.model.name)
print(config.model.depth)
