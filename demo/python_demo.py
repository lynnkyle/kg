import random

import numpy as np
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

# 15. 元组相加 列表相加
ls = ['apple']
tpl = ('apple',)
ls += ['banana']
tpl += ('banana',)
print(ls)
print(tpl)

# 16. update
price = {"apple": 34, "banana": 66, "pear": 78}
price.update({"orange": 1001})
print(price)

# 17. collections.Counter
from collections import Counter

lst = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'a', 'b']
counter = Counter(lst)
print(counter)
print(counter.most_common(2))

# 18. sorted
dic = {'a': 2, 'b': 6, 'c': 4, 'd': 5}
res = sorted(dic, key=dic.get, reverse=True)
print(res)

# 19. scipy.sparse
## 1). sp.coo_matrix
import scipy.sparse as sp

data = np.array([0, 1, 2, 6])
row = np.array([0, 1, 2, 0])
col = np.array([0, 1, 2, 2])
coo = sp.coo_matrix((data, (row, col)), shape=(3, 3))
print(coo)
print(coo.toarray())
sparse_mx = coo.tocoo().astype(np.float32)
print(sparse_mx)
print(sparse_mx.row)
print(sparse_mx.col)
print(sparse_mx.data)

### torch.sparse
sparse_matrix = torch.sparse_coo_tensor(torch.LongTensor([[0, 1, 2], [0, 1, 2]]), torch.FloatTensor([1, 2, 3]),
                                        size=(3, 3))
print(sparse_matrix)

## 2). sp.diags
# data = np.array([1, 2, 3, 4])
# matrix = sp.diags(data, offsets=0)
# print(matrix)
# print(matrix.toarray())

# 20.sort sortargs
lst = [5, 20, 14, 37, 28, 56]
res = np.sort(lst)
print(res)
res = np.argsort(lst)
print(res)

# 21.random
import random

in_edge = [0, 1, 2, 3, 4]
out_edge = [7, 8, 9, 10, 11]
# 错误写法
random.shuffle((in_edge + out_edge))

# 22.Pool 线程池
from multiprocessing import Pool


def square(n):
    return n * n


with Pool(processes=4) as pool:
    result = pool.map(square, range(10))

print(result)

# 23. partial 固定某些参数
from functools import partial


def power(n, x):
    return x ** n


exp = partial(power, x=2)
print(exp(10))
