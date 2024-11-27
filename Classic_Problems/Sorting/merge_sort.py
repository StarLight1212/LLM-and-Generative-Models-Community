def merge_sort(arr):
    """
    归并排序函数
    :param arr: 待排序的列表
    :return: 排序后的列表
    """
    if len(arr) <= 1:  # 如果列表长度小于等于1，直接返回
        return arr

    mid = len(arr) // 2  # 找到中间索引
    left_half = merge_sort(arr[:mid])  # 递归排序左半部分
    right_half = merge_sort(arr[mid:])  # 递归排序右半部分

    return merge(left_half, right_half)  # 合并已排序的两部分

def merge(left, right):
    """
    合并两个已排序的列表
    :param left: 左侧已排序列表
    :param right: 右侧已排序列表
    :return: 合并后的已排序列表
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
    |   如果列表长度 <= 1 |
    |       返回列表      |
    +---------------------+
              |
              v
    +---------------------+
    |   找到中间索引 mid  |
    +---------------------+
              |
              v
    +---------------------+
    |   递归排序左半部分  |
    |   left_half = merge_sort(arr[:mid]) |
    +---------------------+
              |
              v
    +---------------------+
    |   递归排序右半部分  |
    |   right_half = merge_sort(arr[mid:]) |
    +---------------------+
              |
              v
    +---------------------+
    |   合并已排序的两部分 |
    |   return merge(left_half, right_half) |
    +---------------------+
              |
              v
    +---------------------+
    |   合并函数 merge    |
    |   合并两个已排序的列表 |
    +---------------------+
              |
              v
    +---------------------+
    |   返回合并后的列表  |
    +---------------------+
              |
              v
    +---------------------+
    |   结束              |
    +---------------------+
    """
    merged = []  # 存放合并后的结果
    i = j = 0  # 初始化两个指针

    # 合并两个列表
    while i < len(left) and j < len(right):
        if left[i] < right[j]:  # 比较两个列表的当前元素
            merged.append(left[i])  # 将较小的元素添加到合并列表
            i += 1  # 移动左侧指针
        else:
            merged.append(right[j])  # 将较小的元素添加到合并列表
            j += 1  # 移动右侧指针

    # 添加剩余元素
    merged.extend(left[i:])  # 添加左侧剩余元素
    merged.extend(right[j:])  # 添加右侧剩余元素

    return merged  # 返回合并后的列表

# 测试归并排序
if __name__ == "__main__":
    sample_list = [38, 27, 43, 3, 9, 82, 10]  # 待排序的列表
    print("原始列表:", sample_list)
    sorted_list = merge_sort(sample_list)  # 调用归并排序函数
    print("排序后的列表:", sorted_list)
