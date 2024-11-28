"""
双向链表是一种数据结构，每个节点包含指向前一个节点和下一个节点的指针。在本示例中，我们将实现一个简单的双向链表，并结合一个实战应用：管理一个图书馆的书籍列表，支持在任意位置插入和删除书籍。
"""

class Node:
    """双向链表节点"""
    def __init__(self, data):
        self.data = data
        self.previous = None
        self.next = None

    def __str__(self):
        return f"{self.data}"


class DoublyLinkedList:
    """双向链表类"""
    def __init__(self):
        self.head = None
        self.tail = None

    def __iter__(self):
        """迭代链表节点"""
        node = self.head
        while node:
            yield node.data
            node = node.next

    def __str__(self):
        """返回链表的字符串表示"""
        return "->".join([str(item) for item in self])

    def __len__(self):
        """返回链表的长度"""
        return sum(1 for _ in self)

    def insert_at_head(self, data):
        """在链表头部插入新节点"""
        self.insert_at_nth(0, data)

    def insert_at_tail(self, data):
        """在链表尾部插入新节点"""
        self.insert_at_nth(len(self), data)

    def insert_at_nth(self, index: int, data):
        """在指定位置插入新节点"""
        length = len(self)
        if not 0 <= index <= length:
            raise IndexError("list index out of range")
        new_node = Node(data)
        if self.head is None:  # 空链表
            self.head = self.tail = new_node
        elif index == 0:  # 插入头部
            new_node.next = self.head
            self.head.previous = new_node
            self.head = new_node
        elif index == length:  # 插入尾部
            self.tail.next = new_node
            new_node.previous = self.tail
            self.tail = new_node
        else:  # 中间插入
            temp = self.head
            for _ in range(index):
                temp = temp.next
            new_node.previous = temp.previous
            new_node.next = temp
            temp.previous.next = new_node
            temp.previous = new_node

    def delete_head(self):
        """删除头节点"""
        return self.delete_at_nth(0)

    def delete_tail(self):
        """删除尾节点"""
        return self.delete_at_nth(len(self) - 1)

    def delete_at_nth(self, index: int):
        """删除指定位置的节点"""
        length = len(self)
        if not 0 <= index < length:
            raise IndexError("list index out of range")
        if length == 1:  # 只有一个节点
            self.head = self.tail = None
        elif index == 0:  # 删除头节点
            self.head = self.head.next
            if self.head:  # 更新头节点的前驱
                self.head.previous = None
        elif index == length - 1:  # 删除尾节点
            delete_node = self.tail
            self.tail = self.tail.previous
            self.tail.next = None
        else:  # 中间删除
            temp = self.head
            for _ in range(index):
                temp = temp.next
            delete_node = temp
            temp.next.previous = temp.previous
            temp.previous.next = temp.next
        return delete_node.data

    def is_empty(self):
        """检查链表是否为空"""
        return len(self) == 0


# 实战应用示例：管理图书馆的书籍列表
if __name__ == "__main__":
    """
    本示例展示如何使用双向链表管理图书馆的书籍列表，支持在任意位置插入和删除书籍。
    """
    library = DoublyLinkedList()
    
    # 添加书籍
    library.insert_at_tail("《活着》")
    library.insert_at_tail("《百年孤独》")
    library.insert_at_tail("《三体》")
    
    print("当前书籍列表:")
    print(library)  # 输出: 《活着》->《百年孤独》->《三体》

    # 在头部添加书籍
    library.insert_at_head("《小王子》")
    print("添加书籍后:")
    print(library)  # 输出: 《小王子》->《活着》->《百年孤独》->《三体》

    # 删除书籍
    library.delete_head()
    print("删除头部书籍后:")
    print(library)  # 输出: 《活着》->《百年孤独》->《三体》
