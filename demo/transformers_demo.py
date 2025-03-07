from transformers import AutoModel, AutoTokenizer, AutoProcessor

"""
    Bert模型、分词器加载
"""
model = AutoModel.from_pretrained('../pre_trained/bert')
print(model)

tokenizer = AutoTokenizer.from_pretrained('../pre_trained/bert')
print(tokenizer('I am a student'))
print(type(tokenizer))
vacab = tokenizer.get_vocab()
all_token_ids = list(vacab.values())
min_token_id = min(all_token_ids)
max_token_id = max(all_token_ids)
print(len(all_token_ids))

"""
    Beit模型、分词器加载
"""
model = AutoModel.from_pretrained('../pre_trained/beit')
print(model)

# process = AutoProcessor.from_pretrained('../pre_trained/beit')
# print(process)
