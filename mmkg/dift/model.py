import os
import torch
from torch import nn
from transformers import LlamaForCausalLM
from transformers.activations import ACT2FN

"""
    知识图谱模型中得到预训练的嵌入embedding
"""


class EmbeddingModel(nn.Module):
    def __init__(self, emb_dir, emb_dim, intermediate_dim, output_dim, hidden_act):
        """
        :param emb_dir: 嵌入所在的文件
        :param emb_dim: 嵌入的维度
        :param intermediate_dim: adapter隐藏层的维度
        :param output_dim:  adapter输出层的维度
        :param hidden_act:  adapter激活函数
        """
        super().__init__()
        ent_emb_path = os.path.join(emb_dir, 'entity_embeddings.pt')
        query_emb_path = os.path.join(emb_dir, 'query_embeddings.pt')

        ent_emb = torch.load(ent_emb_path)
        ent_emb.requires_grad = False
        self.ent_emb = nn.Embedding.from_pretrained(ent_emb)

        query_emb = torch.load(query_emb_path)
        query_emb.requires_grad = False
        self.query_emb = nn.Embedding.from_pretrained(query_emb)

        self.adapter = nn.Sequential(
            nn.Linear(emb_dim, intermediate_dim),
            ACT2FN[hidden_act],
            nn.Linear(intermediate_dim, output_dim)
        )

        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, query_ids, entity_ids):
        """
        :param query_ids: 查询id
        :param entity_ids: 实体id
        :return:
        """
        query_embeds = self.adapter(self.query_emb(query_ids))
        ent_embeds = self.adapter(self.ent_emb(entity_ids))
        return query_embeds, ent_embeds


"""
    融合知识图谱嵌入(kge)与大语言模型(llm)架构
"""


class KGELlama(nn.Module):
    def __init__(self, tokenizer, llama_model, kge_model):
        """
        :param tokenizer:
        :param llama_model:
        :param kge_model:
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.kge_model = kge_model
        self.llama_model = llama_model

    def forward(self, input_ids, attention_mask, labels, query_ids, entity_ids):
        """
        :param input_ids: 文本输入的token id(含占位符[query]和[entity])
        :param attention_mask: 注意力mask(控制哪些token被关注)
        :param labels: 训练时的目标输出
        :param query_ids: 每条文本对应的查询id
        :param entity_ids: 每条文本对应的k个实体id
        :return:
        """
        query_holder = self.tokenizer.convert_tokens_to_ids(['[QUERY]'])[0]
        entity_holder = self.tokenizer.convert_tokens_to_ids(['[ENTITY]'])[0]
        query_position = torch.nonzero(input_ids == query_holder)
        entity_position = torch.nonzero(input_ids == entity_holder)

        # ??? view(-1)的原因
        query_embeds, entity_embeds = self.kge_model(query_ids, entity_ids.view(-1))

        input_ids[input_ids == query_holder] = self.tokenizer.pad_token_id
        input_ids[input_ids == entity_holder] = self.tokenizer.pad_token_id
        input_emb = self.llama_model.model.embed_tokens(input_ids).clone()
        # ???
        input_emb[query_position[:, 0], query_position[:, 1]] = query_embeds
        input_emb[entity_position[:, 0], entity_position[:, 1]] = entity_embeds

        # 训练/计算损失/微调 把输入送入模型并返回logits和loss
        return self.llama_model(
            input_embeds=input_emb,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate(self, input_ids, query_ids, entity_ids, generation_config):
        query_holder = self.tokenizer.convert_tokens_to_ids(['[QUERY]'])[0]
        entity_holder = self.tokenizer.convert_tokens_to_ids(['[ENTITY]'])[0]
        query_position = torch.nonzero(input_ids == query_holder)
        entity_position = torch.nonzero(input_ids == entity_holder)

        # ??? view(-1)的原因
        query_embeds, entity_embeds = self.kge_model(query_ids, entity_ids.view(-1))

        input_ids[input_ids == query_holder] = self.tokenizer.pad_token_id
        input_ids[input_ids == entity_holder] = self.tokenizer.pad_token_id
        input_emb = self.llama_model.model.model.embed_tokens(input_ids).clone()
        # ???
        input_emb[query_position[:, 0], query_position[:, 1]] = query_embeds
        input_emb[entity_position[:, 0], entity_position[:, 1]] = entity_embeds

        # 生成文本 基于输入生成下一个token序列
        return self.llama_model.generate(
            input_embeds=input_emb,
            generation_config=generation_config
        )

    def save_pretrained(self, peft_model_path):
        self.llama_model.save_pretrained(peft_model_path)
        torch.save(self.kge_model.state_dict(), os.path.join(os.path.dirname(peft_model_path), 'kge_model.pth'))


if __name__ == '__main__':
    model = LlamaForCausalLM.from_pretrained('models--TheBloke--Llama-2-7B-fp16')
    res = model.model.embed_tokens(torch.LongTensor([[1, 2, 3], [4, 5, 6]]))
    print(res)