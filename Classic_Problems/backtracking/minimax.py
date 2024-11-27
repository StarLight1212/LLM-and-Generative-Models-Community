from __future__ import annotations
import math

def minimax(
    depth: int, node_index: int, is_max: bool, scores: list[int], height: float
) -> int:
    """
    实现Minimax算法，帮助在两人游戏中获得最佳分数。
    如果当前玩家是最大化者，则分数被最大化；如果是最小化者，则分数被最小化。

    参数:
    - depth: 当前在游戏树中的深度。
    - node_index: 当前节点在分数列表中的索引。
    - is_max: 布尔值，指示当前移动是最大化者（True）还是最小化者（False）。
    - scores: 包含游戏树叶子节点分数的列表。
    - height: 游戏树的最大高度。

    返回:
    - 当前玩家的最佳分数。

    >>> import math
    >>> scores = [90, 23, 6, 33, 21, 65, 123, 34423]
    >>> height = math.log(len(scores), 2)
    >>> minimax(0, 0, True, scores, height)
    65
    >>> minimax(-1, 0, True, scores, height)
    Traceback (most recent call last):
        ...
    ValueError: Depth cannot be less than 0
    >>> minimax(0, 0, True, [], 2)
    Traceback (most recent call last):
        ...
    ValueError: Scores cannot be empty
    >>> scores = [3, 5, 2, 9, 12, 5, 23, 23]
    >>> height = math.log(len(scores), 2)
    >>> minimax(0, 0, True, scores, height)
    12
    """

    # 检查深度是否有效
    if depth < 0:
        raise ValueError("Depth cannot be less than 0")
    # 检查分数列表是否为空
    if len(scores) == 0:
        raise ValueError("Scores cannot be empty")

    # 基本情况：如果当前深度等于树的高度，返回当前节点的分数
    if depth == height:
        return scores[node_index]

    # 如果是最大化者的回合，选择两个可能移动中的最大分数
    if is_max:
        return max(
            minimax(depth + 1, node_index * 2, False, scores, height),  # 左子树
            minimax(depth + 1, node_index * 2 + 1, False, scores, height)  # 右子树
        )

    # 如果是最小化者的回合，选择两个可能移动中的最小分数
    return min(
        minimax(depth + 1, node_index * 2, True, scores, height),  # 左子树
        minimax(depth + 1, node_index * 2 + 1, True, scores, height)  # 右子树
    )

def main() -> None:
    """
    主函数，计算并打印使用Minimax算法的最佳值。
    """
    # 示例分数和高度计算
    scores = [90, 23, 6, 33, 21, 65, 123, 34423]
    height = math.log(len(scores), 2)  # 计算树的高度

    # 计算并打印最佳值
    print("Optimal value : ", end="")
    print(minimax(0, 0, True, scores, height))

if __name__ == "__main__":
    import doctest  # 导入doctest模块用于测试
    doctest.testmod()  # 运行文档测试，验证函数的正确性
    main()  # 调用主函数
