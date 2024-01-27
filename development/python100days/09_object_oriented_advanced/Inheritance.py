# 继承

class Person(object):
    def __init__(self,name,age):
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
        print(f'{self._name} is playing!!')

    def watch_tv(self):
        if self._age < 18:
            print(f'{self._name} can only watch cartoon!')
        else:
            print(f'{self._name} can watch whatever he wants to watch!')

class Student(Person):
    def __init__(self,name,age,grade):
        super().__init__(name,age)
        self._grade = grade

    @property
    def grade(self):
        return self._grade

    @grade.setter
    def grade(self,grade):
        self._grade = grade

    def study(self,course):
        print(f'{self._name} is studying {course}!!')

class Teacher(Person):
    def __init__(self,name,age,title):
        super().__init__(name,age)
        self._title = title

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self,title):
        self._title=title

    def teacher(self,course):
        print(f'{self._name} is teaching {course}!!')

def main():
    stu = Student('Jack',15,'first year of high school')
    stu.study('math')
    stu.watch_tv()
    teacher = Teacher('Einstein',50,'physicist')
    teacher.teacher('physics')
    teacher.watch_tv()

if __name__ == '__main__':
    main()



