class Person(object):
    def __init__(self,name,age):
        self._name = name
        self._age = age

    # 使用@property包装器来包装getter和setter方法，使得对属性的访问既安全又方便
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
    print(p1.name)
    print(p1.age)
    p1.play()
    p1.age = 18
    p1.play()
    # p1.name = 'lucas' #AttributeError: can't set attribute 'name'

if __name__ == "__main__":
    main()