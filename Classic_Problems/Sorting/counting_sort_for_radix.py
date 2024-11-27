def counting_sort_for_radix(arr, exp):
    """
    基数排序中的计数排序函数
    :param arr: 待排序的列表
    :param exp: 当前位数的指数（1, 10, 100, ...）
    :return: 排序后的列表
    +---------------------+
    |   开始              |
    +---------------------+
              |
              v
    +---------------------+
    |   输入待排序列表    |
    +---------------------+
              |
              v
    +---------------------+
    |   找到最大值 max_val |
    +---------------------+
              |
              v
    +---------------------+
    |   exp = 1          |
    +---------------------+
              |
              v
    +---------------------+
    |   while max_val // exp > 0 |
    +---------------------+
              |
              v
    +---------------------+
    |   调用计数排序函数  |
    |   counting_sort_for_radix(arr, exp) |
    +---------------------+
              |
              v
    +---------------------+
    |   exp *= 10        |
    +---------------------+
              |
              v
    +---------------------+
    |   返回排序后的列表  |
    +---------------------+
              |
              v
    +---------------------+
    |   结束              |
    +---------------------+
    """
    n = len(arr)  # 获取列表长度
    output = [0] * n  # 创建输出数组
    count = [0] * 10  # 创建计数数组，范围为0-9

    # 统计每个数字在当前位的出现次数
    for i in range(n):
        index = (arr[i] // exp) % 10  # 获取当前位的数字
        count[index] += 1  # 计数

    # 计算计数数组的累积和
    for i in range(1, 10):
        count[i] += count[i - 1]

    # 从后向前填充输出数组
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10  # 获取当前位的数字
        output[count[index] - 1] = arr[i]  # 放入输出数组
        count[index] -= 1  # 减少计数

    return output  # 返回排序后的数组

def radix_sort(arr):
    """
    基数排序函数
    :param arr: 待排序的列表
    :return: 排序后的列表
    """
    max_val = max(arr)  # 找到最大值
    exp = 1  # 从最低位开始

    # 对每一位进行计数排序
    while max_val // exp > 0:
        arr = counting_sort_for_radix(arr, exp)  # 进行计数排序
        exp *= 10  # 移动到下一位

    return arr  # 返回排序后的列表

# 测试基数排序
if __name__ == "__main__":
    sample_list = [170, 45, 75, 90, 802, 24, 2, 66]  # 待排序的列表
    print("原始列表:", sample_list)
    sorted_list = radix_sort(sample_list)  # 调用基数排序函数
    print("排序后的列表:", sorted_list)
