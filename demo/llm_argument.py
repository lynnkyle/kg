from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, LlamaForCausalLM, LlamaTokenizer


# 1. HfArgumentParser
@dataclass
class MyArguments:
    model_name: str = field(default='KGEllama', metadata={})
    learing_rate: float = field(default=5e-5, metadata={})
    batch_size: int = field(default=32, metadata={})


parser = HfArgumentParser(MyArguments)
args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
print(args, _)


# 2
@dataclass
class GenerationArguments:
    # 设置output长度
    max_new_tokens: Optional[int] = field(default=64, metadata={'help': 'Max New Tokens'})
    min_new_tokens: Optional[int] = field(default=1, metadata={'help': 'Min New Tokens'})

    # 设置生成策略(贪心搜索, 不惩罚相似内容)
    do_sample: Optional[bool] = field(default=True, metadata={'help': 'Sample Or Not'})
    num_beams: Optional[int] = field(default=1, metadata={'help': 'Num Beam'})
    num_beam_groups: Optional[int] = field(default=1, metadata={'help': 'Num Beam Groups'})
    penalty_alpha: Optional[float] = field(default=None, metadata={'help': 'Penalty Alpha'})
    use_cache: Optional[bool] = field(default=True, metadata={'help': 'Use Cache Or Not'})

    # 设置词概率处理(增强多样性, 禁止重复)
    temperature: Optional[float] = field(default=1.0, metadata={'help': 'Temperature'})
    top_k: Optional[int] = field(default=50, metadata={'help': 'Top K'})
    top_p: Optional[float] = field(default=0.9, metadata={'help': 'Top p'})
    typical_p: Optional[float] = field(default=1.0, metadata={'help': 'Typical p'})
    diversity_penalty: Optional[float] = field(default=0.0, metadata={'help': 'Diversity Penalty'})
    repetition_penalty: Optional[float] = field(default=1.0, metadata={'help': 'Repetition Penalty'})
    length_penalty: Optional[float] = field(default=1.0, metadata={'help': 'Length Penalty'})
    no_repeat_ngram_size: Optional[int] = field(default=0, metadata={})

    # 设置输出格式
    num_return_sequences: Optional[int] = field(default=1, metadata={'help': 'Num Return Sequences'})
    output_scores: Optional[bool] = field(default=False, metadata={'help': 'Return Output Scores Or Not'})
    return_dict_in_generate: Optional[bool] = field(default=True, metadata={
        'help': 'Return Dict(Sequences、 Scores、 Beam_Indices、 Sequence_Scores) In Generate Or Not'})


parser = HfArgumentParser(GenerationArguments)
generation_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
print("generation_args==>", generation_args, _)

# 3. LlamaForCausalLM, LlamaTokenizer
llama_tokenizer = LlamaTokenizer.from_pretrained('/home/ps/lzy/subaru/models--TheBloke--Llama-2-7B-fp16')

llama_tokenizer.add_tokens('[QUERY]')
llama_tokenizer.add_tokens('[QUERY]')
llama_tokenizer.add_tokens('[QUERY]')
llama_tokenizer.add_tokens('[ENTITY]')

print(llama_tokenizer.convert_tokens_to_ids(['[QUERY]'][0]))
print(llama_tokenizer.convert_tokens_to_ids(['[ENTITY]'][0]))

# generate 与 input_ids、 input_embeds
prompt = 'The capital of France is'
inputs = llama_tokenizer(prompt, return_tensors='pt').to("cuda")
llama_model = LlamaForCausalLM.from_pretrained('/home/ps/lzy/subaru/models--TheBloke--Llama-2-7B-fp16',
                                               device_map="auto")
output_ids = llama_model.generate(
    input_ids=inputs.input_ids,
    max_new_tokens=20,
    do_sample=True,
    top_p=0.95,
    temperature=0.8
)
print(output_ids)
print(llama_tokenizer.decode(output_ids[0], skip_special_tokens=True))
input_embeds = llama_model.model.embed_tokens(inputs.input_ids).clone()
output_embs = llama_model.generate(
    inputs_embeds=input_embeds,
    max_new_tokens=20,
    do_sample=True,
    top_p=0.95,
    temperature=0.8
)
print(llama_tokenizer.decode(output_embs[0], skip_special_tokens=True))
