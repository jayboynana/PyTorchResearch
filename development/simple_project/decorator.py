def my_decorator(func):
    def wrapper():
        print('1')
        func()
        print('2')
    return wrapper

@my_decorator
def myprint():
    print('test')

def myprint2():
    print('testing')

if __name__ == "__main__":
    myprint()
    myprint2()
