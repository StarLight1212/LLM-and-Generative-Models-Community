"""
链表是一种动态数据结构，由一系列节点组成，每个节点包含数据和指向下一个节点的指针。在本示例中，我们将实现一个简单的链表，并结合一个实战应用：管理一个学生的成绩单，找出中间成绩。
"""

from __future__ import annotations


class Node:
    """链表节点"""
    def __init__(self, data: int) -> None:
        self.data = data
        self.next = None


class LinkedList:
    """链表类"""
    def __init__(self):
        self.head = None

    def push(self, new_data: int) -> int:
        """在链表头部添加新节点并返回新节点的数据"""
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node
        return self.head.data

    def middle_element(self) -> int | None:
        """
        找到链表的中间元素
        >>> link = LinkedList()
        >>> link.middle_element()
        No element found.
        >>> link.push(5)
        5
        >>> link.push(6)
        6
        >>> link.push(8)
        8
        >>> link.push(10)
        10
        >>> link.push(12)
        12
        >>> link.push(17)
        17
        >>> link.push(7)
        7
        >>> link.push(3)
        3
        >>> link.push(20)
        20
        >>> link.push(-20)
        -20
        >>> link.middle_element()
        12
        """
        slow_pointer = self.head
        fast_pointer = self.head
        
        if not self.head:
            print("No element found.")
            return None
        
        # 使用快慢指针法找到中间元素
        while fast_pointer and fast_pointer.next:
            fast_pointer = fast_pointer.next.next
            slow_pointer = slow_pointer.next
        
        return slow_pointer.data


# 实战应用示例：管理学生成绩单
if __name__ == "__main__":
    """
    本示例展示如何使用链表管理学生的成绩单，并找出中间成绩。
    """
    link = LinkedList()
    n = int(input("请输入成绩数量: ").strip())
    print("请输入每个成绩:")
    for _ in range(n):
        data = int(input().strip())
        link.push(data)
    
    middle_score = link.middle_element()
    if middle_score is not None:
        print(f"中间成绩是: {middle_score}")
