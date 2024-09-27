def solve_n_queens_dfs(n):
    def is_safe(board, row, col):
        for r in range(row):
            if board[r] == col or abs(board[r] - col) == abs(r - row):
                return False
        return True

    def dfs(board, row):
        # 如果成功放置完所有皇后，记录解法
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col  # 放置皇后
                dfs(board, row + 1)  # 递归下一行
                board[row] = -1  # 回溯

    solutions = []
    board = [-1] * n  # 初始空棋盘
    dfs(board, 0)  # 从第0行开始
    return solutions

# 例子：解决8皇后问题
n = 8
dfs_solutions = solve_n_queens_dfs(n)
print(f"DFS找到 {len(dfs_solutions)} 个解法")
