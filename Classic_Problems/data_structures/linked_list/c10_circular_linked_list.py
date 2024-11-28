"""
双向循环链表是一种数据结构，每个节点包含指向前一个节点和下一个节点的指针，并且最后一个节点指向第一个节点。在本示例中，我们将实现一个简单的循环链表，并结合一个实战应用：管理一个音乐播放列表，支持在任意位置插入和删除歌曲。
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
class CircularLinkedList:
    """循环链表类"""
    head: Node | None = None  # 指向头节点
    tail: Node | None = None  # 指向尾节点

    def __iter__(self) -> Iterator[Any]:
        """迭代链表节点"""
        node = self.head
        if node is not None:
            while True:
                yield node.data
                node = node.next_node
                if node == self.head:
                    break

    def __len__(self) -> int:
        """返回链表的长度"""
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        """返回链表的字符串表示"""
        return "->".join(str(item) for item in self)

    def insert_tail(self, data: Any) -> None:
        """在链表尾部插入新节点"""
        self.insert_nth(len(self), data)

    def insert_head(self, data: Any) -> None:
        """在链表头部插入新节点"""
        self.insert_nth(0, data)

    def insert_nth(self, index: int, data: Any) -> None:
        """在指定位置插入新节点"""
        if index < 0 or index > len(self):
            raise IndexError("list index out of range.")
        new_node: Node = Node(data)
        if self.head is None:  # 空链表
            new_node.next_node = new_node  # 第一个节点指向自己
            self.head = self.tail = new_node
        elif index == 0:  # 插入头部
            new_node.next_node = self.head
            self.tail.next_node = new_node
            self.head = new_node
        else:  # 中间或尾部插入
            temp: Node = self.head
            for _ in range(index - 1):
                temp = temp.next_node
            new_node.next_node = temp.next_node
            temp.next_node = new_node
            if index == len(self):  # 更新尾节点
                self.tail = new_node

    def delete_front(self) -> Any:
        """删除并返回头节点的数据"""
        return self.delete_nth(0)

    def delete_tail(self) -> Any:
        """删除并返回尾节点的数据"""
        return self.delete_nth(len(self) - 1)

    def delete_nth(self, index: int) -> Any:
        """删除指定位置的节点并返回其数据"""
        if not 0 <= index < len(self):
            raise IndexError("list index out of range.")
        assert self.head is not None
        delete_node: Node = self.head
        if self.head == self.tail:  # 只有一个节点
            self.head = self.tail = None
        elif index == 0:  # 删除头节点
            self.head = self.head.next_node
            self.tail.next_node = self.head
        else:  # 中间或尾部删除
            for _ in range(index - 1):
                delete_node = delete_node.next_node
            temp = delete_node.next_node
            delete_node.next_node = temp.next_node
            if index == len(self) - 1:  # 删除尾节点
                self.tail = delete_node
        return temp.data

    def is_empty(self) -> bool:
        """检查链表是否为空"""
        return len(self) == 0


# 实战应用示例：管理音乐播放列表
if __name__ == "__main__":
    """
    本示例展示如何使用循环链表管理音乐播放列表，支持在任意位置插入和删除歌曲。
    """
    playlist = CircularLinkedList()
    
    # 添加歌曲
    playlist.insert_tail("Song A")
    playlist.insert_tail("Song B")
    playlist.insert_tail("Song C")
    
    print("当前播放列表:")
    print(playlist)  # 输出: Song A->Song B->Song C

    # 在头部添加歌曲
    playlist.insert_head("Song D")
    print("添加歌曲后:")
    print(playlist)  # 输出: Song D->Song A->Song B->Song C

    # 删除歌曲
    playlist.delete_front()
    print("删除头部歌曲后:")
    print(playlist)  # 输出: Song A->Song B->Song C
