from collections import deque

def solve_n_queens_bfs(n):
    def is_safe(board, row, col):
        for r in range(row):
            if board[r] == col or abs(board[r] - col) == abs(r - row):
                return False
        return True

    queue = deque([[]])  # 队列初始化为空棋盘状态
    solutions = []

    while queue:
        board = queue.popleft()
        row = len(board)

        if row == n:
            solutions.append(board)
            continue

        for col in range(n):
            if is_safe(board, row, col):
                new_board = board + [col]  # 放置皇后
                queue.append(new_board)  # 将新的状态加入队列

    return solutions

# 例子：解决8皇后问题
n = 8
bfs_solutions = solve_n_queens_bfs(n)
print(f"BFS找到 {len(bfs_solutions)} 个解法")
