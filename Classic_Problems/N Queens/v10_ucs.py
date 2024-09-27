import heapq

def solve_n_queens_ucs(n):
    def is_safe(board, row, col):
        for r in range(row):
            if board[r] == col or abs(board[r] - col) == abs(r - row):
                return False
        return True

    # 最小堆初始化为空棋盘状态，代价为0
    heap = [(0, [])]
    solutions = []

    while heap:
        cost, board = heapq.heappop(heap)
        row = len(board)

        if row == n:
            solutions.append(board)
            continue

        for col in range(n):
            if is_safe(board, row, col):
                new_board = board + [col]
                heapq.heappush(heap, (cost + 1, new_board))  # 每步的代价+1

    return solutions

# 例子：解决8皇后问题
n = 8
ucs_solutions = solve_n_queens_ucs(n)
print(f"UCS找到 {len(ucs_solutions)} 个解法")
