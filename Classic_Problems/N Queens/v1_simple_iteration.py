import json
import time
import numpy as np
from typing import List, Tuple, Dict


def create_env_and_save_seq(file_name='seq.json'):
    start = time.time()

    # 生成每行与每列都只有一个皇后的序列
    seq = [[i, j, k, l, m, n, o, p]
           for i in range(1, 9)
           for j in range(1, 9)
           for k in range(1, 9)
           for l in range(1, 9)
           for m in range(1, 9)
           for n in range(1, 9)
           for o in range(1, 9)
           for p in range(1, 9)
           if all([i != j, i != k, i != l, i != m, i != n, i != o, i != p,
                   j != k, j != l, j != m, j != n, j != o, j != p,
                   k != l, k != m, k != n, k != o, k != p,
                   l != m, l != n, l != o, l != p,
                   m != n, m != o, m != p,
                   n != o, n != p,
                   o != p])]  # 筛选出【每行与每列都只有一个皇后】的序列

    print(f'有 {len(seq)} 个可能的序列')

    # 将序列存储到指定文件
    with open(file_name, 'w') as file_object:
        json.dump(seq, file_object)

    end = time.time()

    print('Successful!')
    print(f'已将生成的序列存储到文件 {file_name} 中，用时 {end - start:.2f}s')
    return seq


# 暴力穷举
def simple_iteration_solution(seqs: List, N: int):
    solutions = 0
    width_size = N + 1
    for s in seqs:
        # 改变为9*9二维数组，为了后面方便使用，只用后八行和后八列的8*8部分，作为一个空白棋盘
        a = np.array([0]*width_size*width_size).reshape(width_size, width_size)
        # 假设当前序列对应的棋盘满足条件
        flag = 1

        # 根据序列，从第一列到最后一列的顺序，在对应位置放一个皇后，生成当前序列对应的棋盘
        for i in range(1, 9):
            a[s[i-1]][i] = 1

        # 检查当前序列的八个皇后在各自的两条对角线上是否有其他皇后
        for i in range(1, 9):
            t1 = t2 = s[i - 1]
            # 看左半段
            for j in range(i-1, 0, -1):
                if t1 != 1:
                    t1 -= 1
                    """
                    1 0 0 0 0
                    0 1 0 0 0
                    0 0 1 0 0
                    0 0 0 a 0
                    'a' as the starter
                    """
                    if a[t1][j] == 1:
                        # 正对角线左半段上有其他皇后，表示当前序列不满足条件，不用再检查次对角线左半段、正对角线右半段、次对角线右半段
                        flag = 0
                        break

                if t2 != 8:
                    t2 += 1
                    """
                    0 0 0 0 a
                    0 0 0 1 0
                    0 0 1 0 0
                    0 1 0 0 0
                    'a' as the starter
                    """
                    if a[t2][j] == 1:
                        # 次对角线左半段上有其他皇后，表示当前序列不满足条件，不用再检查正对角线右半段、次对角线右半段
                        flag = 0
                        break

            if flag == 0:
                break  # 当前序列不满足条件，不用再检查正对角线右半段、次对角线右半段

            t1 = t2 = s[i-1]
            # 看右半段
            for j in range(i+1, 9):
                if t1 != 1:
                    t1 -= 1
                    """
                    0 0 0 0 1
                    0 0 0 1 0
                    0 0 1 0 0
                    0 a 0 0 0
                    'a' as the starter
                    """
                    if a[t1][j] == 1:
                        # 正对角线右半段上有其他皇后，表示当前序列不满足条件，不用再检查次对角线右半段
                        flag = 0
                        break

                    if t2 != 8:
                        t2 += 1
                        """
                        a 0 0 0 0
                        0 1 0 0 0
                        0 0 1 0 0
                        0 0 0 1 0
                        'a' as the starter
                        """
                        if a[t2][j] == 1:
                            # 次对角线右半段上有其他皇后，表示当前序列不满足条件
                            flag = 0
                            break

        if flag == 1:  # 经过层层筛选，如果序列符合条件则执行下面内容
            solutions += 1  # 计数+1
            print('第' + str(solutions) + '个解，此序列：' + str(s) + ' 符合条件，对应棋盘如下：')

            for i in a[1:]:  # 输出对应棋盘
                for j in i[1:]:
                    print(j, ' ', end="")  # 有了end=""，print就不会换行
                print()  # 输出完一行后再换行，这里不能是print('\n')，否则会换两行

            print('---------------')  # 分割线
        print('共' + str(solutions) + '个解')  # 最后再明确一下有几个解


# 调用函数
seqs = create_env_and_save_seq()
simple_iteration_solution(seqs, 8)
