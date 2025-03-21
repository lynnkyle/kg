import requests
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor,BeitImageProcessor
"""
    1.Bert模型、分词器加载
"""
# config
config = AutoConfig.from_pretrained('../pre_trained/bert')
print(config)
# model
model = AutoModel.from_pretrained('../pre_trained/bert')
print("bert_token_size_0==>", model.config.vocab_size)  # [30522]
# tokenizer
tokenizer = AutoTokenizer.from_pretrained('../pre_trained/bert')
vocab = tokenizer.get_vocab()
print("bert_token_size_1==>", len(vocab))  # [30522]
feature = tokenizer("Hello world", return_tensors="pt")
print(feature['input_ids'].shape)

"""
    2.Beit模型、分词器加载
"""
# config
config = AutoConfig.from_pretrained('../pre_trained/beit')
print(config)
# model
model = AutoModel.from_pretrained('../pre_trained/beit')
print("beit_token_size_0==>", model.config.vocab_size)  # [8192]
codebook_embeddings = model.embeddings.patch_embeddings.projection.weight
print("beit_token_size_1==>", codebook_embeddings.shape)
# processor
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
processor = AutoProcessor.from_pretrained('../pre_trained/beit')
feature = processor(images=image, return_tensors='pt')
print(feature['pixel_values'].shape)
