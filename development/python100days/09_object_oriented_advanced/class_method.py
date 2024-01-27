from time import time, localtime, sleep

class Clock(object):

    def __init__(self, hour=0, minute=0, second=0):
        self._hour = hour
        self._minute = minute
        self._second = second

    # 类方法的第一个参数约定名为cls，它代表的是当前类相关的信息的对象（类本身也是一个对象，有的地方也称之为类的元数据对象），
    # 通过这个参数我们可以获取和类相关的信息并且可以创建出类的对象
    @classmethod
    def now(cls):
        ctime = localtime(time())
        return cls(ctime.tm_hour,ctime.tm_min,ctime.tm_sec)  # 可以创建对象

    def run(self):
        self._second += 1
        if self._second == 60:
            self._second = 0
            self._minute += 1
            if self._minute == 60:
                self._minute = 0
                self._hour += 1
                if self._hour == 24:
                    self._hour = 0

    def show(self):
        return '%02d:%02d:%02d' % (self._hour, self._minute, self._second)

def main():
    clock = Clock.now()
    while True:
        print(clock.show())
        sleep(1)
        clock.run()
    # print(clock.show())
    # sleep(1)
    # clock.run()
    # print(clock.show())

if __name__ == '__main__':
    main()