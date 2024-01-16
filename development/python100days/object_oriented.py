class Student(object):

    def __init__(self,name,age):
        self.name = name
        self.age = age

    def study(self,course_name):
        print(f'{self.name} is studying {course_name}!')

    def watch_movie(self):
        if self.age<18:
            print(f'{self.name} can only watch cartoon！')
        else:
            print(f'{self.name} can watch whatever he wants!')

def main():
    stu1 = Student('kakarot',20)
    stu1.study('Python')
    stu1.watch_movie()

if __name__ == "__main__":
    main()