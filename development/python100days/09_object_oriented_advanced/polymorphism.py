# 多态

from abc import ABCMeta, abstractmethod

class Pet(object,metaclass=ABCMeta):
    def __init__(self,nickname):
        self._nickname = nickname

    @abstractmethod
    def make_voice(self):
        pass
    
class Dog(Pet):
    def make_voice(self):
        print(f'{self._nickname} is barking!!')

class Cat(Pet):
    def make_voice(self):
        print(f'{self._nickname} is Meowing!!')

def main():
    pets = [Dog('John'), Cat('Smith'),Dog('James')]
    for pet in pets:
        pet.make_voice()
    # dog = Dog('John')
    # dog.make_voice()
    # cat = Cat('Smith')
    # cat.make_voice()

if __name__ == '__main__':
    main()
    # pet = Pet('P') # 如果一个类中存在抽象方法那么这个类就不能够实例化（创建对象）