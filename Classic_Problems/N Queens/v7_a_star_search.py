import heapq


def heuristic(board):
    # Heuristic: number of attacking queen pairs (only for the queens already placed)
    attacking_pairs = 0
    for i in range(len(board)):  # Only loop through placed queens
        for j in range(i + 1, len(board)):  # Compare queens that are already placed
            if board[i] == board[j] or abs(board[i] - board[j]) == j - i:
                attacking_pairs += 1
    return attacking_pairs


def solve_n_queens_a_star_all_solutions(n):
    def get_neighbors(board):
        # Generate neighbors by placing a queen in each column at different rows
        neighbors = []
        row = len(board)  # Current row to place the next queen
        if row == n:
            return neighbors  # If all queens are placed, no more neighbors

        for col in range(n):
            if is_safe(board, row, col):
                neighbors.append(board + [col])  # Add new queen in this column

        return neighbors

    def is_safe(board, row, col):
        # Check if placing a queen at (row, col) is safe
        for r, c in enumerate(board):
            if c == col or abs(row - r) == abs(col - c):
                return False
        return True

    initial_board = []  # Start with an empty board
    heap = [(0, initial_board)]  # Heuristic value, and board state
    solutions = []
    visited = set()

    while heap:
        h, current_board = heapq.heappop(heap)

        if tuple(current_board) in visited:
            continue

        visited.add(tuple(current_board))

        # If we've placed all queens and heuristic is 0 (no attacking queens), it's a solution
        if len(current_board) == n and h == 0:
            solutions.append(current_board)
            continue

        # Generate neighbors (next possible placements of queens)
        for neighbor in get_neighbors(current_board):
            if tuple(neighbor) not in visited:
                # Push neighbors to the heap, heuristic is attacking pairs count
                heapq.heappush(heap, (heuristic(neighbor), neighbor))

    return solutions


# Example usage:
n = 8
all_solutions = solve_n_queens_a_star_all_solutions(n)
print(f"Found {len(all_solutions)} solutions for {n}-Queens.")
for solution in all_solutions:
    print(solution)
