"""
    1. 判断文件是否相同
"""
import filecmp

file1 = "index1.txt"
file2 = "index2.txt"

if filecmp.cmp(file1, file2, shallow=False):
    print("两个文件相同")
else:
    print("两个文件不同")

"""
    2. 判断文件第几行不同
"""

with open(file1, "r") as f1, open(file2, "r") as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    for i, (lin1, lin2) in enumerate(zip(lines1, lines2)):
        if lin1 != lin2:
            print(i)
