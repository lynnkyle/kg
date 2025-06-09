from dataclasses import dataclass, field
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

# 2. LlamaForCausalLM, LlamaTokenizer
llama_tokenizer = LlamaTokenizer.from_pretrained('/home/ps/lzy/subaru/models--TheBloke--Llama-2-7B-fp16')

llama_tokenizer.add_tokens('[QUERY]')
llama_tokenizer.add_tokens('[QUERY]')
llama_tokenizer.add_tokens('[QUERY]')
llama_tokenizer.add_tokens('[ENTITY]')

print(llama_tokenizer.convert_tokens_to_ids(['[QUERY]'][0]))
print(llama_tokenizer.convert_tokens_to_ids(['[ENTITY]'][0]))
