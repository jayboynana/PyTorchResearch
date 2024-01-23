from math import sqrt

class Triangle(object):
    def __init__(self,a,b,c):
        self._a = a
        self._b = b
        self._c = c

    # 先写一个方法来验证三条边长是否可以构成三角形，
    # 这个方法很显然就不是对象方法，因为在调用这个方法时三角形对象尚未创建出来（因为都不知道三条边能不能构成三角形），
    # 所以这个方法是属于三角形类而并不属于三角形对象的
    @staticmethod
    def is_valid_triangle(a,b,c):
        return a+b > c and a+c > b and b+c > a

    def perimeter(self):
        return self._a + self._b + self._c

    def area(self):
        half = self.perimeter()/2
        return sqrt((half * (half - self._a) * (half - self._b) * (half - self._c)))

def main():
    a,b,c = 3,4,5
    # 静态方法和类方法都是通过给类发消息来调用的
    if Triangle.is_valid_triangle(a,b,c):
        t = Triangle(a,b,c)
        print(t.perimeter())
        print(t.area())
    else:
        print('invalid triangle!')

if __name__ == '__main__':
    main()