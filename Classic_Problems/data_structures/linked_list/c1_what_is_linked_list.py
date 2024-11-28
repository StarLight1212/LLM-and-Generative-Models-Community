"""
链表是一种数据结构，由一系列节点组成，每个节点包含数据和指向下一个节点的指针。链表的优点是可以动态地增加或减少节点，适合需要频繁插入和删除操作的场景。

链表的基本操作包括：
1. 插入节点
2. 删除节点
3. 查找节点
4. 遍历链表

在本示例中，我们将实现一个简单的链表，并结合一个实战应用：管理一个图书馆的书籍列表。
"""

from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class Node:
    """链表节点"""
    data: Any
    next_node: Node | None = None


@dataclass
class LinkedList:
    """链表"""
    head: Node | None = None

    def __iter__(self) -> Iterator:
        """返回链表的迭代器"""
        node = self.head
        while node:
            yield node.data
            node = node.next_node

    def push(self, new_data: Any) -> None:
        """在链表头部添加新节点"""
        new_node = Node(new_data)
        new_node.next_node = self.head
        self.head = new_node

    def remove(self, data: Any) -> None:
        """删除链表中指定数据的节点"""
        current = self.head
        previous = None
        while current and current.data != data:
            previous = current
            current = current.next_node
        if current is None:  # 数据未找到
            return
        if previous is None:  # 删除的是头节点
            self.head = current.next_node
        else:
            previous.next_node = current.next_node

    def display(self) -> None:
        """打印链表中的所有节点数据"""
        print("链表内容:", list(self))


# 实战应用示例：管理图书馆的书籍列表
if __name__ == "__main__":
    """
    本示例展示如何使用链表管理图书馆的书籍列表。
    """
    library = LinkedList()
    
    # 添加书籍
    library.push("《活着》")
    library.push("《百年孤独》")
    library.push("《三体》")
    library.push("《小王子》")
    
    print("当前图书馆书籍列表:")
    library.display()
    
    # 删除一本书
    library.remove("《三体》")
    print("删除《三体》后的书籍列表:")
    library.display()

### 代码说明：
# 1. 链表结构：定义了`Node`和`LinkedList`类，分别表示链表的节点和链表本身。
# 2. 基本操作：实现了插入（`push`）和删除（`remove`）节点的功能，以及遍历链表的功能（`display`）。
# 3. 实战应用：通过一个图书馆的书籍管理示例，展示如何使用链表来动态管理书籍列表。用户可以添加书籍并删除指定书籍，体现了链表的灵活性。
