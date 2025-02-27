import random
import torch

# 1. dict的使用
triples = torch.tensor([[1, 0, 4],
                        [1, 0, 5],
                        [2, 1, 6],
                        [2, 1, 4],
                        [3, 2, 5],
                        [3, 2, 8]])
print(triples)

from collections import defaultdict

dic_set_1 = defaultdict(set)
dic_set_1[0].add(4)
dic_set_1[0].add(5)
dic_set_2 = defaultdict(set)
dic_set_2[1].add(6)
dic_set_2[1].add(4)
dic_set_3 = defaultdict(set)
dic_set_3[2].add(5)
dic_set_3[2].add(8)
dic = {1: dic_set_1, 2: dic_set_2, 3: dic_set_3}
print(dic)
print(dic[1][0])

dic = {'a': 1, 'b': 2, 'c': 3}
for k in dic:
    print("k===>", k, dic[k])
ls = dic.keys()
print("ls==>", ls)

# 2. random.sample / random.shuffle
n = 5
ls = random.sample(range(5), 5)
print(ls)

idx = [_ for _ in range(n)]
random.shuffle(idx)
print(idx)

# 3. *
ls = ['apple', 'pear', 'egg']
print(*ls)
