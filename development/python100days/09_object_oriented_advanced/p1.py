# Ultraman vs monster


from abc import ABC, abstractmethod
from random import randint,randrange

class fighter(object):

    __slots__ = ('_name', '_hp')
    def __init__(self,name,hp):
        self._name = name
        self._hp = hp

    @property
    def name(self):
        return self._name

    @property
    def hp(self):
        return self._hp

    @hp.setter
    def hp(self,hp):
        self._hp = hp if hp > 0 else 0

    @property
    def alive(self):
        return self._hp > 0

    @abstractmethod
    def attack(self,other):
        """
        Args:
            other:要攻击的对象
        """
        pass

class Ultraman(fighter):

    __slots__ = ['_name','_hp','_magic']
    def __init__(self,name,hp,magic):
        super().__init__(name,hp)
        self._magic =magic

    def attack(self,other):
        injury = randint(10,20)
        other._hp -= injury
        print(f'普通攻击对{other._name}造成伤害{injury}!')
        return True

    def huge_attack(self,other):
        if self._magic > 50 :
            self._magic -= 50
            injury = other._hp * 3 // 4
            injury = injury if injury > 50 else 50
            other._hp -= injury
            print(f'大招对{other._name}造成伤害{injury}!')
            return True
        else:
            print("魔法值不足，只能普通攻击了！！")
            self.attack(other)
            return False

    def group_attack(self,others):
        if self._magic > 20:
            self._magic -= 20
            for other in others:
                if other.alive:
                    injury = randint(15,25)
                    other._hp -= injury
                    print(f'群体攻击对{other._name}造成伤害{injury}')
            return True
        else:
            print("魔法值不足以施放群体攻击！！")
            return False

    def resume_magic(self):
        incr_magic = randint(10,20)
        self._magic += incr_magic
        return incr_magic

    def __str__(self):
        return f'Ultraman {self._name} \n hp: {self._hp} \n magic: {self._magic}'

class Monster(fighter):
    __slots__ = ['_name', '_hp']
    def attack(self,other):
        injury = randint(5,10)
        other._hp -= injury
        print(f'{self._name}对{other._name}造成伤害{injury}')

    def __str__(self):
        return f'Monster {self._name} \n hp: {self._hp}'

def display_info(ultraman,monsters):
    print(ultraman)
    for monster in monsters:
        print(monster)

def main():
    u = Ultraman('泰罗',1000,100)
    m1 = Monster('Smith',200)
    m2 = Monster('Jason',300)
    ms = [m1,m2]
    display_info(u,ms)
    print('======进行对战!!======')
    u.attack(m1)
    u.huge_attack(m2)
    u.group_attack(ms)
    m1.attack(u)
    m2.attack(u)
    print('======对战结束!!======')
    display_info(u,ms)

if __name__ == '__main__':
    main()