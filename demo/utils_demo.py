"""
    1. 判断文件是否相同
"""
# import filecmp
#
# file1 = "index1.txt"
# file2 = "index2.txt"
#
# if filecmp.cmp(file1, file2, shallow=False):
#     print("两个文件相同")
# else:
#     print("两个文件不同")

"""
    2. 判断文件第几行不同
"""

# with open(file1, "r") as f1, open(file2, "r") as f2:
#     lines1 = f1.readlines()
#     lines2 = f2.readlines()
#     for i, (lin1, lin2) in enumerate(zip(lines1, lines2)):
#         if lin1 != lin2:
#             print(i)

"""
    3. easydict设置config
"""

# from easydict import EasyDict
#
# config = EasyDict({
#     'name': 'lzy',
#     'age': 24,
#     'city': 'fuzhou'
# })
# config.sex = '男'
# print(config['name'])
# print(config['age'])
# print(config['city'])

"""
    3. networkx图论与网络分析库, 用于创建、操作和研究复杂网络结构
"""
import networkx as nx

# 创建图
# g = nx.Graph()
# # g = nx.MultiGraph(g) 多重图
# # g = nx.MultiDiGraph(g) 有向多重图
# g.add_node(1)
# g.add_nodes_from([2, 3])
# g.add_edge(1, 2, relation=3)
# g.add_edge(2, 3, relation=1)
# g.add_edge(3, 1, relation=2)
# g.add_edges_from([(2, 3), (3, 1)])

# 遍历图
# print(g)
# print("所有节点", g.nodes())
# print("所有边", g.edges())
# for u, v, data in g.edges(data=True):
#     print(f"{u} -> {v}: relation = {data['relation']}")
#
# # 计算最短路径
# path = nx.shortest_path(g, source=1, target=3)
# print("从1到3的最短路径:", path)

# 可视化图
# import matplotlib.pyplot as plt
#
# nx.draw(g, with_labels=True)
# plt.show()

"""
    4. networkx图论的使用
"""
# g = nx.DiGraph()
# g.add_edge(1, 2, relation=1)
# g.add_edge(1, 3, relation=1)
# g.add_edge(1, 4, relation=1)
# g.add_edge(2, 3, relation=2)
# g.add_edge(3, 4, relation=3)
# print(g.out_edges(1))
# print(g.out_edges(1, data=True))

"""
    5. json
"""
import json

file_path = "/home/ps/lzy/kg/data/benchmark/DB15K/entity.json"  # 替换成你的 JSON 文件路径

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        json.load(f)
    print("JSON 文件格式正确")
except json.JSONDecodeError as e:
    print("JSON 文件格式错误")
    print(f"错误信息：{e}")
    print(f"错误位置：第 {e.lineno} 行，第 {e.colno} 列，字符偏移 {e.pos}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        error_line = lines[e.lineno - 1]

        print("\n出错行内容：")
        print(error_line.strip())

        # 显示错误列位置用 ^ 指示
        pointer_line = " " * (e.colno - 1) + "^"
        print(pointer_line)