# 练习1：定义一个类描述数字时钟

from time import sleep

class Clock(object):
    def __init__(self,hours,mins,seconds):
        self._hours = hours
        self._mins = mins
        self._seconds = seconds

    def run(self):
        self._seconds += 1
        if self._seconds == 60:
            self._seconds = 0
            self._mins += 1
            if self._mins == 60:
                self._mins = 0
                self._hours += 1
                if self._hours == 24:
                    self._hours = 0

    def show(self):
        print(f'it is {self._hours:02}:{self._mins:02}:{self._seconds:02} now!')
        print('{:02}:{:02}:{:02}'.format(self._hours,self._mins,self._seconds))

def main():
    clock = Clock(22,59,59)
    print(clock.show())
    sleep(1)
    clock.run()
    print(clock.show())

if __name__ == "__main__":
    main()