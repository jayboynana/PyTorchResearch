# 练习2：定义一个类描述平面上的点并提供移动点和计算到另一个点距离的方法。

from math import sqrt

class Point(object):

    def __init__(self,init_x,init_y):
        self.x = init_x
        self.y = init_y

    def move_to(self,x,y):
        self.x = x
        self.y = y

    def move_by(self,dx,dy):
        self.x += dx
        self.y += dy

    def compute_distance(self,point2):
        offset_x = abs(self.x - point2.x)
        offset_y = abs(self.y - point2.y)
        dis = sqrt(offset_x**2 + offset_y**2)
        print(f'The distance between two points is {dis:0.2f}')

def main():
    p1 = Point(1,1)
    p2 = Point(3,4)
    p1.compute_distance(p2)
    p2.compute_distance(p1)
    p1.move_to(2,2)
    p1.compute_distance(p2)
    p1.move_by(-1,-1)
    p1.compute_distance(p2)

if __name__ == "__main__":
    main()
