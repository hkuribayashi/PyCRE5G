class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return '[x={}, y={}, z={}]'.format(self.x, self.y, self.z)
