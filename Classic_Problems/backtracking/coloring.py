def valid_coloring(
    neighbours: list[int], colored_vertices: list[int], color: int
) -> bool:
    """
    检查当前颜色是否有效
    :param neighbours: 当前顶点的邻居列表
    :param colored_vertices: 已着色的顶点列表
    :param color: 当前尝试的颜色
    :return: 如果当前颜色有效返回True，否则返回False

    >>> neighbours = [0, 1, 0, 1, 0]
    >>> colored_vertices = [0, 2, 1, 2, 0]
    >>> valid_coloring(neighbours, colored_vertices, 1)
    True
    >>> valid_coloring(neighbours, colored_vertices, 2)
    False
    """
    # 检查是否有邻居使用相同的颜色
    return not any(
        neighbour == 1 and colored_vertices[i] == color
        for i, neighbour in enumerate(neighbours)
    )


def util_color(
    graph: list[list[int]], max_colors: int, colored_vertices: list[int], index: int
) -> bool:
    """
    递归尝试为图的每个顶点着色
    :param graph: 图的邻接矩阵
    :param max_colors: 最大可用颜色数
    :param colored_vertices: 当前顶点的颜色列表
    :param index: 当前处理的顶点索引
    :return: 如果成功着色返回True，否则返回False

    >>> graph = [[0, 1, 0, 0, 0],
    ...          [1, 0, 1, 0, 1],
    ...          [0, 1, 0, 1, 0],
    ...          [0, 1, 1, 0, 0],
    ...          [0, 1, 0, 0, 0]]
    >>> max_colors = 3
    >>> colored_vertices = [0, 1, 0, 0, 0]
    >>> util_color(graph, max_colors, colored_vertices, 3)
    True
    >>> max_colors = 2
    >>> util_color(graph, max_colors, colored_vertices, 3)
    False
    """
    # 基本情况：如果所有顶点都已处理，返回True
    if index == len(graph):
        return True

    # 递归步骤：尝试为当前顶点着色
    for color in range(max_colors):
        if valid_coloring(graph[index], colored_vertices, color):
            # 为当前顶点着色
            colored_vertices[index] = color
            # 递归调用，处理下一个顶点
            if util_color(graph, max_colors, colored_vertices, index + 1):
                return True
            # 回溯：如果没有找到有效的着色，重置当前顶点的颜色
            colored_vertices[index] = -1
    return False  # 如果没有找到有效的着色，返回False


def color(graph: list[list[int]], max_colors: int) -> list[int]:
    """
    主函数，调用util_color进行图的着色
    :param graph: 图的邻接矩阵
    :param max_colors: 最大可用颜色数
    :return: 如果成功着色，返回着色列表；否则返回空列表

    >>> graph = [[0, 1, 0, 0, 0],
    ...          [1, 0, 1, 0, 1],
    ...          [0, 1, 0, 1, 0],
    ...          [0, 1, 1, 0, 0],
    ...          [0, 1, 0, 0, 0]]
    >>> max_colors = 3
    >>> color(graph, max_colors)
    [0, 1, 0, 2, 0]
    >>> max_colors = 2
    >>> color(graph, max_colors)
    []
    """
    # 初始化所有顶点的颜色为-1（表示未着色）
    colored_vertices = [-1] * len(graph)

    # 调用递归函数尝试着色
    if util_color(graph, max_colors, colored_vertices, 0):
        return colored_vertices  # 返回成功着色的结果

    return []  # 如果无法着色，返回空列表


if __name__ == "__main__":
    # 示例图的邻接矩阵
    graph = [[0, 1, 0, 0, 0],
             [1, 0, 1, 0, 1],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 0, 0, 0]]
    
    max_colors = 3
    result = color(graph, max_colors)
    print("着色结果:", result)  # 输出着色结果

    max_colors = 2
    result = color(graph, max_colors)
    print("着色结果:", result)  # 输出着色结果
