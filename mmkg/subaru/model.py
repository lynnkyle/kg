import torch
from torch import nn


class KnowledgePrompting(nn.Module):
    def __init__(self, model, kge_model='data/transe.pt', pretrain_emb_path=None, adapter_type='mlp'):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        pretrain_embeddings = torch.load(open(kge_model, 'rb'))
        if pretrain_emb_path is None:
            self.embeddings = PretrainKGEmbedding(
                pretrain_ent_embs=pretrain_embeddings,
                dim_llm=4096,
                adapter_type=adapter_type
            )
        else:
            self.embeddings = torch.load(pretrain_emb_path)

    def forward(self, input_ids, position_ids, past_key_values, labels, use_cache, output_attentions,
                output_hidden_states, return_dict, embedding_ids):
        """
        :param input_ids:
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
        token_embeds = self.model.model.embed_tokens(input_ids)  # 大模型提取的特征
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)
        prefix_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)
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


class PretrainKGEmbedding(nn.Module):
    def __init__(self, pretrain_ent_embs, dim_llm, num_prefix=1, adapter_type='mlp'):
        super().__init__()
