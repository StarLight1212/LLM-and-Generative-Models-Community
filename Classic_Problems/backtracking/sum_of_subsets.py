from __future__ import annotations

def generate_sum_of_subsets_soln(nums: list[int], max_sum: int) -> list[list[int]]:
    """
    生成所有和为给定值M的子集。

    参数:
    - nums: 输入的非负整数列表。
    - max_sum: 目标和M。

    返回:
    - list[list[int]]: 所有满足条件的子集列表。
    """
    result: list[list[int]] = []  # 存储结果的列表
    path: list[int] = []  # 当前路径（子集）
    num_index = 0  # 当前数字的索引
    remaining_nums_sum = sum(nums)  # 剩余数字的总和
    create_state_space_tree(nums, max_sum, num_index, path, result, remaining_nums_sum)
    return result


def create_state_space_tree(
    nums: list[int],
    max_sum: int,
    num_index: int,
    path: list[int],
    result: list[list[int]],
    remaining_nums_sum: int,
) -> None:
    """
    创建状态空间树，使用深度优先搜索（DFS）遍历每个分支。
    当满足以下任一条件时，终止节点的分支：
    1. 当前路径的和大于目标和。
    2. 当前路径的和加上剩余数字的和小于目标和。

    参数:
    - nums: 输入的非负整数列表。
    - max_sum: 目标和M。
    - num_index: 当前数字的索引。
    - path: 当前路径（子集）。
    - result: 存储结果的列表。
    - remaining_nums_sum: 剩余数字的总和。
    """
    # 如果当前路径的和大于目标和，或当前路径的和加上剩余数字的和小于目标和，返回
    if sum(path) > max_sum or (remaining_nums_sum + sum(path)) < max_sum:
        return

    # 如果当前路径的和等于目标和，将路径添加到结果中
    if sum(path) == max_sum:
        result.append(path)
        return

    # 遍历剩余的数字
    for index in range(num_index, len(nums)):
        # 递归调用，选择当前数字并继续搜索
        create_state_space_tree(
            nums,
            max_sum,
            index + 1,  # 选择下一个数字
            path + [nums[index]],  # 将当前数字添加到路径中
            result,
            remaining_nums_sum - nums[index],  # 更新剩余数字的总和
        )


# 示例输入
nums = [3, 34, 4, 12, 5, 2]
max_sum = 9
result = generate_sum_of_subsets_soln(nums, max_sum)

# 打印结果
print("满足条件的子集有:")
for subset in result:
    print(subset)
