def solve_n_queens(n):
    def is_safe(board, row, col):
        # 检查当前列和两条对角线上是否有冲突的皇后
        for r in range(row):
            if board[r] == col or abs(board[r] - col) == abs(r - row):
                return False
        return True

    def solve(board, row):
        # 如果所有皇后都已经成功放置
        if row == n:
            solution = []
            for i in range(n):
                line = '.' * board[i] + 'Q' + '.' * (n - board[i] - 1)
                solution.append(line)
            solutions.append(solution)
            return

        # 尝试在当前行的每一列放置皇后
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col  # 放置皇后
                solve(board, row + 1)  # 尝试放置下一行的皇后
                board[row] = -1  # 回溯，移除皇后

    solutions = []
    board = [-1] * n  # 使用一维数组表示棋盘，初始值为 -1 表示未放置皇后
    solve(board, 0)
    return solutions

# 例子：解决8皇后问题并输出所有解
n = 8
solutions = solve_n_queens(n)
print(f"共找到 {len(solutions)} 个解法：")
for solution in solutions:
    for line in solution:
        print(line)
    print()  # 每个解之间空一行
