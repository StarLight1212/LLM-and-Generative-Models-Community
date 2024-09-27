def solve_n_queens(n):
    def backtrack(row):
        if row == n:
            # 找到一个解，将其加入到结果中
            solutions.append(board[:])
            return

        for col in range(n):
            if cols[col] or diag1[row - col + n - 1] or diag2[row + col]:
                continue  # 如果列或对角线有冲突，跳过

            # 放置皇后
            board[row] = col
            cols[col] = diag1[row - col + n - 1] = diag2[row + col] = True

            # 递归到下一行
            backtrack(row + 1)

            # 回溯，移除皇后
            cols[col] = diag1[row - col + n - 1] = diag2[row + col] = False

    solutions = []
    board = [-1] * n  # 用一维数组表示棋盘，索引是行号，值是列号
    cols = [False] * n  # 每一列是否有皇后
    diag1 = [False] * (2 * n - 1)  # 左上到右下的对角线
    diag2 = [False] * (2 * n - 1)  # 右上到左下的对角线

    backtrack(0)  # 从第0行开始
    return solutions


# 例子：解决8皇后问题
n = 8
all_solutions = solve_n_queens(n)
print(f"找到 {len(all_solutions)} 个解法")
for solution in all_solutions:
    print(solution)
