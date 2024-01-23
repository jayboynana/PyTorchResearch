# __slots__ 魔法

class Person(object):

    # 注意: __slots__的限定只对当前类的对象生效，对子类并不起任何作用
    # 限定Person对象只能绑定_name, _age和_gender属性
    __slots__ = ('_name', '_age','_gender')
    def __init__(self, name, age):
        self._name = name
        self._age = age

    @property
    def name(self):
        return self._name

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self,age):
        self._age = age

    def play(self):
        if self._age <= 16:
            print(f'{self._name}正在玩飞行棋')
        else:
            print(f'{self._name}正在玩斗地主')

def main():
    p1 = Person('jayce',12)
    p1.play()
    p1._gender = 'male'
    print(p1._gender)
    # _is_gay不在slots所指定的范围，尝试绑定会报错
    p1._is_gay = True # AttributeError: 'Person' object has no attribute '_is_gay'

if __name__ == '__main__':
    main()