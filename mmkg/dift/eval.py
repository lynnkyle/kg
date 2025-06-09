import numpy as np
import torch

from mmkg.dift.data import KGDataset
from tqdm import tqdm


class Evaluator(object):
    def __init__(self, args, tokenizer, model, data_module, generation_config):
        self.sample_size = 200
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.data_module = data_module
        self.generation_config = generation_config

    @torch.no_grad()
    def eval_metric(self, dataset: KGDataset):
        self.model.eval()
        preds = []
        raw_ranks = np.array([])
        ranks = np.array([])
        print_step = 1000
        data_num = len(dataset)

        for begin_idx in range(0, data_num, print_step):
            end_idx = min(begin_idx + print_step, data_num)
            generated = []
            for ex_idx, ex in enumerate(tqdm(dataset[begin_idx:end_idx])):
                prompt = ex['input']
                if self.args.model_class == 'KGELlama':
                    inputs = self.tokenizer(prompt, return_tensors='pt')
                    input_ids = inputs['input_ids'].cuda()
                    output = self.model.generate(
                        input_ids=input_ids,
                        query_ids=torch.LongTensor([ex['query_id']]).to(input_ids.device),
                        entity_ids=torch.LongTensor([ex['entity_id']]).to(input_ids.device),
                        generation_config=self.generation_config
                    )
                    generated.append(output.sequences[0].cpu().numpy().tolist())
                else:
                    raise NotImplementedError
                ex.pop('input')
            batch_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for ex_idx, ex in enumerate(dataset[begin_idx:end_idx]):
                target = ex.pop('output')
                rank = ex['rank']
                pred = str(batch_preds[ex_idx]).strip()

                topK_names = ex['topk_names']
                if target == pred:
                    rank = 1
                else:
                    if pred not in set(topK_names) or topK_names.index(pred) >= rank:
                        rank += 1

                # ex['target'] = target
                # ex['pred_rank'] = rank
                # ex['pred'] = pred
                preds.append(ex)
                raw_ranks = np.append(raw_ranks, ex['rank'])
                ranks = np.append(ranks, rank)

            def compute_metrics(rank):
                metrics = {
                    'hits1': np.mean(rank <= 1),
                    'hits3': np.mean(rank <= 3),
                    'hits10': np.mean(rank <= 10),
                    'mrr': np.mean(1. / rank)
                }
                metrics = {k: round(v, 3) for k, v in metrics.items()}

            logger.info('=' * 80)
            compute_metrics(raw_ranks)
            compute_metrics(ranks)

        return preds
