class Node:
    def __init__(self, value, succeeding=None, previous=None):
        self.value = value
        self.succeeding = succeeding
        self.previous = previous


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def push(self, value):
        node = Node(value)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            node.succeeding = self.head
            self.head.previous = node
            self.head = node
        self.length += 1

    def pop(self):
        if self.head is None:
            raise IndexError("List is empty")
        value = self.head.value
        self.head = self.head.succeeding
        if self.head is None:
            self.tail = None
        self.length -= 1
        return value

    def shift(self):
        if self.head is None:
            raise IndexError("List is empty")
        value = self.head.value
        self.head = self.head.succeeding
        if self.head is None:
            self.tail = None
        self.length -= 1
        return value

    def unshift(self, value):
        node = Node(value)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            node.succeeding = self.head
            self.head.previous = node
            self.head = node
        self.length += 1

    def delete(self, value):
        current = self.head
        while current is not None:
            if current.value == value:
                if current.previous is not None:
                    current.previous.succeeding = current.succeeding
                if current.succeeding is not None:
                    current.succeeding.previous = current.previous
                self.length -= 1
                return
            current = current.succeeding
        raise ValueError("Value not found")

    def __len__(self):
        return self.length

    def __iter__(self):
        current = self.head
        while current is not None:
            yield current.value
            current = current.succeeding
