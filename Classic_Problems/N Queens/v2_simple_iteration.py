import json
import time
import numpy as np
from typing import List


def generate_queen_sequences(N: int = 8) -> List[List[int]]:
    """
    生成满足N皇后问题的所有可能棋盘序列。
    :param N: 棋盘大小，默认为8。
    :return: 返回所有满足条件的序列。
    """
    return [[i, j, k, l, m, n, o, p]
            for i in range(1, N+1)
            for j in range(1, N+1)
            for k in range(1, N+1)
            for l in range(1, N+1)
            for m in range(1, N+1)
            for n in range(1, N+1)
            for o in range(1, N+1)
            for p in range(1, N+1)
            if len({i, j, k, l, m, n, o, p}) == N]  # 每行每列均不相同


def save_sequences_to_file(sequences: List[List[int]], file_name: str):
    """
    将序列保存到指定的文件中。
    :param sequences: 需要保存的序列列表。
    :param file_name: 文件名。
    """
    with open(file_name, 'w') as file_object:
        json.dump(sequences, file_object)
    print(f"序列成功保存到 {file_name}.")


def validate_sequence_on_diagonals(sequence: List[int], N: int) -> bool:
    """
    检查序列是否满足对角线没有皇后相互攻击。
    :param sequence: 棋盘皇后的排列序列。
    :param N: 棋盘大小。
    :return: 若满足条件返回True，否则返回False。
    """
    for i in range(N):
        for j in range(i + 1, N):
            if abs(sequence[i] - sequence[j]) == abs(i - j):
                return False
    return True


def solve_n_queens_problem(seqs: List[List[int]], N: int = 8):
    """
    通过简单迭代的方式求解N皇后问题，输出满足条件的解。
    :param seqs: 输入的皇后排列序列。
    :param N: 棋盘大小，默认为8。
    """
    solutions = 0
    for seq in seqs:
        if validate_sequence_on_diagonals(seq, N):
            solutions += 1
            print(f'第 {solutions} 个解，此序列: {seq} 对应棋盘如下:')
            print_chessboard(seq, N)
    print(f'共 {solutions} 个解')


def print_chessboard(sequence: List[int], N: int):
    """
    打印出皇后摆放的棋盘布局。
    :param sequence: 皇后的位置序列。
    :param N: 棋盘大小。
    """
    board = np.zeros((N, N), dtype=int)
    for col, row in enumerate(sequence):
        board[row-1, col] = 1  # 1代表皇后

    for row in board:
        print(" ".join(map(str, row)))
    print('---------------')


def create_env_and_save_seq(file_name: str = 'seq.json', N: int = 8):
    """
    封装整个流程：生成序列、验证并输出结果，同时保存序列到文件。
    :param file_name: 保存文件的名称。
    :param N: 棋盘大小，默认为8。
    """
    start = time.time()

    print(f"正在生成 {N} 皇后问题的序列...")
    sequences = generate_queen_sequences(N)
    save_sequences_to_file(sequences, file_name)

    print(f"开始验证序列的合法性...")
    solve_n_queens_problem(sequences, N)

    end = time.time()
    print(f'任务完成，耗时 {end - start:.2f} 秒')


# 调用函数
create_env_and_save_seq()
