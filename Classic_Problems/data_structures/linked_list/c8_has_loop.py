"""
链表是一种动态数据结构，由一系列节点组成，每个节点包含数据和指向下一个节点的指针。在本示例中，我们将实现一个简单的链表，并结合一个实战应用：检测链表中是否存在循环。
"""

from __future__ import annotations
from typing import Any


class ContainsLoopError(Exception):
    """自定义异常，用于检测循环链表"""
    pass


class Node:
    """链表节点"""
    def __init__(self, data: Any) -> None:
        self.data: Any = data
        self.next_node: Node | None = None

    def __iter__(self):
        """迭代节点，检测循环"""
        node = self
        visited = set()
        while node:
            if node in visited:
                raise ContainsLoopError  # 检测到循环
            visited.add(node)
            yield node.data
            node = node.next_node

    @property
    def has_loop(self) -> bool:
        """
        检测链表是否存在循环
        >>> root_node = Node(1)
        >>> root_node.next_node = Node(2)
        >>> root_node.next_node.next_node = Node(3)
        >>> root_node.next_node.next_node.next_node = Node(4)
        >>> root_node.has_loop
        False
        >>> root_node.next_node.next_node.next_node = root_node.next_node
        >>> root_node.has_loop
        True
        """
        try:
            list(self)  # 尝试迭代节点
            return False
        except ContainsLoopError:
            return True


# 实战应用示例：检测链表中的循环
if __name__ == "__main__":
    """
    本示例展示如何使用链表检测循环，模拟一个简单的任务管理系统。
    """
    # 创建一个链表
    root_node = Node(1)
    root_node.next_node = Node(2)
    root_node.next_node.next_node = Node(3)
    root_node.next_node.next_node.next_node = Node(4)

    # 检测循环
    print("检测链表是否存在循环:")
    print(root_node.has_loop)  # 输出: False

    # 创建循环
    root_node.next_node.next_node.next_node = root_node.next_node
    print("检测链表是否存在循环:")
    print(root_node.has_loop)  # 输出: True

    # 创建另一个链表
    root_node = Node(5)
    root_node.next_node = Node(6)
    root_node.next_node.next_node = Node(5)  # 这里没有形成循环
    root_node.next_node.next_node.next_node = Node(6)
    print("检测链表是否存在循环:")
    print(root_node.has_loop)  # 输出: False

    # 创建单个节点的链表
    root_node = Node(1)
    print("检测链表是否存在循环:")
    print(root_node.has_loop)  # 输出: False
