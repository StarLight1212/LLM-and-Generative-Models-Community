"""
链表是一种动态数据结构，由一系列节点组成，每个节点包含数据和指向下一个节点的指针。在本示例中，我们将实现一个简单的链表，并结合一个实战应用：管理一个学生的成绩单，并按指定组大小反转节点顺序。
"""

from __future__ import annotations
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass
class Node:
    """链表节点"""
    data: int
    next_node: Node | None = None


class LinkedList:
    """链表类"""
    def __init__(self, ints: Iterable[int]) -> None:
        """初始化链表并添加元素"""
        self.head: Node | None = None
        for i in ints:
            self.append(i)

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

    def append(self, data: int) -> None:
        """在链表尾部添加新节点"""
        if not self.head:
            self.head = Node(data)
            return
        node = self.head
        while node.next_node:
            node = node.next_node
        node.next_node = Node(data)

    def reverse_k_nodes(self, group_size: int) -> None:
        """
        按指定组大小反转链表中的节点
        """
        if self.head is None or self.head.next_node is None:
            return

        length = len(self)
        dummy_head = Node(0)
        dummy_head.next_node = self.head
        previous_node = dummy_head

        while length >= group_size:
            current_node = previous_node.next_node
            next_node = current_node.next_node
            for _ in range(1, group_size):
                current_node.next_node = next_node.next_node
                next_node.next_node = previous_node.next_node
                previous_node.next_node = next_node
                next_node = current_node.next_node
            previous_node = current_node
            length -= group_size
        self.head = dummy_head.next_node


# 实战应用示例：管理学生成绩单
if __name__ == "__main__":
    """
    本示例展示如何使用链表管理学生的成绩单，并按指定组大小反转节点顺序。
    """
    ll = LinkedList([85, 90, 78, 92, 88])  # 学生成绩列表
    print(f"原始成绩单链表: {ll}")
    k = 2  # 组大小
    ll.reverse_k_nodes(k)
    print(f"反转每组大小为 {k} 的成绩单链表: {ll}")
