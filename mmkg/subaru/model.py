import torch
from torch import nn
from transformers import LlamaForCausalLM

"""
    将原有模型（如 LLaMA）前面插入一段来自知识图谱实体的嵌入（kg_embeds），与原始 token 嵌入拼接后送入大语言模型进行训练或推理。
"""


class KnowledgePrompting(nn.Module):
    def __init__(self, model, kge_model='data/transe.pt', pretrain_emb_path=None, adapter_type='mlp'):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        if kge_model is not None:
            pretrain_embeddings = torch.load(open(kge_model, 'rb'))
        if pretrain_emb_path is None:
            self.embeddings = PretrainKGEmbedding(
                pretrain_ent_embs=pretrain_embeddings,
                dim_llm=4096,
                adapter_type=adapter_type
            )
        else:
            self.embeddings = torch.load(pretrain_emb_path)
            # self.embeddings = pretrain_emb_path

    def forward(self, input_ids=None, position_ids=None, past_key_values=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, embedding_ids=None):
        """
        :param input_ids: token ID
        :param position_ids:
        :param past_key_values:
        :param labels:
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :param embedding_ids:
        :return:
        """
        kg_embeds = self.embeddings(embedding_ids)
        batch_size, seq_len, _ = kg_embeds.shape
        token_embeds = self.model.model.embed_tokens(input_ids)  # 将输入的token ID映射为对应的词向量嵌入, 整数->连续向量
        input_embeds = torch.cat((kg_embeds, token_embeds),
                                 dim=1)  # [batch_size, kg_seq_len + prompt_seq_len, hidden_size]
        prefix_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        new_labels = torch.cat((prefix_labels, labels), dim=-1)
        return self.model(
            input_ids=None,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


"""
    将预训练的知识图谱实体嵌入（pretrain_ent_embs）通过适配器（adapter）转换为可与大语言模型（如 LLaMA）的输入嵌入拼接的高维向量，主要用于提示（prompt）增强。
"""


class PretrainKGEmbedding(nn.Module):
    def __init__(self, pretrain_ent_embs, llm_dim, num_prefix=1, adapter_type='mlp'):
        super().__init__()
        self.num_prefix = num_prefix
        self.llm_dim = llm_dim
        self.emb_dim = num_prefix * llm_dim
        self.embeddings = nn.Embedding.from_pretrained(pretrain_ent_embs)
        self.pretrain_dim = self.embeddings.weight.shape[1]
        self.embeddings.requires_grad_(False)
        self.adapter_type = adapter_type
        if adapter_type == 'fc':
            self.adapter = nn.Linear(self.pretrain_dim, self.emb_dim)
        elif adapter_type == 'mlp':
            self.adapter = nn.Sequential(
                nn.Linear(self.pretrain_dim, 3 * self.emb_dim),
                nn.ReLU(),
                nn.Linear(3 * self.emb_dim, self.emb_dim)
            )
        else:
            raise NotImplementedError

    def forward(self, triple_ids):
        """
        :param triple_ids:  [batch_size, num_token]
        :return:
        """
        batch_size = triple_ids.shape[0]
        num_token = triple_ids.shape[1]
        ent = triple_ids.reshape(-1, num_token)
        with torch.no_grad():
            emb = self.embeddings(ent)
        prefix = self.adapter(emb).reshape(batch_size, -1, self.llm_dim)
        return prefix


if __name__ == '__main__':
    # KnowledgePrompting的测试
    model = LlamaForCausalLM.from_pretrained('models--TheBloke--Llama-2-7B-fp16', torch_dtype=torch.float16)
    embedding = nn.Embedding(4, 4096, dtype=torch.float16)
    prompt = KnowledgePrompting(model, None, embedding)
    result = prompt(input_ids=torch.tensor([[0, 1, 2]]),
                    labels=torch.tensor([[0, 1, 2]]),
                    embedding_ids=torch.tensor([[0, 1, 2]]))
    print(result)
