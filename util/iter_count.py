import sys

class IterCount(object):
    def  __init__(self, msg):
        self.count = 0
        self.msg = msg
        self.length = len(msg) + 6

    def __del__(self):
        print('\nDone!\n')

    def update(self):
        self.count += 1
        self.print_msg()

    def print_msg(self):
        sys.stdout.write(self.msg.format(self.count))
        sys.stdout.write('\b' * self.length)
        sys.stdout.flush()
