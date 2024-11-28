"""
算法：将两个已排序的链表合并为一个排序的链表。

在本示例中，我们将实现一个简单的排序链表，并结合一个实战应用：管理一个学生的成绩单，将两个不同班级的成绩合并为一个排序的成绩单。
"""

from __future__ import annotations
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# 测试数据
test_data_odd = (3, 9, -11, 0, 7, 5, 1, -1)
test_data_even = (4, 6, 2, 0, 8, 10, 3, -2)


@dataclass
class Node:
    """链表节点"""
    data: int
    next_node: Node | None = None


class SortedLinkedList:
    """排序链表类"""
    def __init__(self, ints: Iterable[int]) -> None:
        """初始化排序链表并添加元素"""
        self.head: Node | None = None
        for i in sorted(ints):  # 按升序排序
            self.head = Node(i, self.head)

    def __iter__(self) -> Iterator[int]:
        """返回链表的迭代器"""
        node = self.head
        while node:
            yield node.data
            node = node.next_node

    def __len__(self) -> int:
        """返回链表的长度"""
        return sum(1 for _ in self)

    def __str__(self) -> str:
        """返回链表的字符串表示"""
        return " -> ".join([str(node) for node in self])


def merge_lists(sll_one: SortedLinkedList, sll_two: SortedLinkedList) -> SortedLinkedList:
    """
    合并两个排序链表并返回一个新的排序链表
    """
    return SortedLinkedList(list(sll_one) + list(sll_two))


# 实战应用示例：管理学生成绩单
if __name__ == "__main__":
    """
    本示例展示如何使用排序链表管理学生的成绩单，并将两个班级的成绩合并为一个排序的成绩单。
    """
    SSL = SortedLinkedList
    class_a_grades = [85, 90, 78, 92, 88]  # 班级A的成绩
    class_b_grades = [80, 95, 70, 88, 91]  # 班级B的成绩

    # 创建两个排序链表
    sorted_class_a = SSL(class_a_grades)
    sorted_class_b = SSL(class_b_grades)

    # 合并两个班级的成绩
    merged_grades = merge_lists(sorted_class_a, sorted_class_b)
    print("合并后的排序成绩单:")
    print(merged_grades)  # 输出合并后的成绩单
