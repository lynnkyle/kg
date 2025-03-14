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

# 4. logging
import logging

#       设置方式一
logging.basicConfig(level=logging.DEBUG)
logging.debug('debug')
logging.info('info')
logging.warning('warning')
logging.error('error')
logging.critical('critical')

#       设置方式二
logger = logging.getLogger("vista")
logger.setLevel(logging.INFO)
format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(format)
logger.addHandler(handler)
logger.info('debug')
logger.warning('warning')
logger.error('error')


# 5. vars 返回对象的__dict__属性
class People(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age


people = People('People', 20)
print(vars(people))

# 6. tqdm
import time
from tqdm import tqdm

# ls = ['apple', 'banana', 'pear', 'orange']
# for s in tqdm(ls):
#     print(s)
#     time.sleep(2)

# with tqdm(desc='fruit list', total=len(ls), unit='g') as bar:
#     i = 0
#     while i < len(ls):
#         print(ls[i])
#         time.sleep(3)
#         bar.update(1)
#         i = i + 1

# 7. 索引
x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])  # [2,3,3]
index = torch.tensor([[False, False, True], [False, False, True]])  # [2,3]
print(x[index])

# 8. min
min_num = min([-1000, 100, 8])
print(min_num)

dc = {'a': 1, 'b': 2, 'c': 3}
print(len(dc))
print(dc.keys())
print(dc.values())
print(dc.items())

# 9. Counter
from collections import Counter

data = ['apple', 'banana', 'apple', 'apple', 'banana', 'pear', 'pear']
res = Counter(data)
ans = res.most_common(2)
print(ans)

# 10. sorted
ls = [7, 56, 2, 54, 65, 99, 34]
res = sorted(ls, key=lambda x: -x)
print(res)

# 11. 列表
ls = defaultdict(list)
ls[2].append(3)
ls[3].append(4)
ls[3].append(5)
print(ls)

# 12 assert 断言
a = 10
b = 2
assert a % b == 0
print(a % b)

# 13. isinstance
ls = [1, 2, 3]
print(isinstance(ls, list))

# 14. 字典转换为对象
from torch import nn

ACL2CLS = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}

from collections import OrderedDict


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2FN = ClassInstantier(ACL2CLS)
relu = ACT2FN['relu']
print(relu)
