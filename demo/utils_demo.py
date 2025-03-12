import filecmp

file1 = "tensor1.txt"
file2 = "tensor2.txt"

if filecmp.cmp(file1, file2, shallow=False):
    print("两个文件相同")
else:
    print("两个文件不同")