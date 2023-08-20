def overwrite(self, data):
    if self.buffer[self.end] is not None:
        self.buffer[self.end] = data
        self.end = (self.end + 1) % self.capacity
        if self.buffer[self.start] is not None:
            self.start = (self.start + 1) % self.capacity
    else:
        self.write(data)