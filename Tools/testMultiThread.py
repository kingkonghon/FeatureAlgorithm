from multiprocessing import Pool
import time

con = {'a':1, 'b':2, 'c':3, 'd':4}


class test_class:
    def __init__(self, **kwargs):
        self.a = kwargs.get('a')
        self.b = kwargs.get('b')
        self.c = kwargs.get('c')
        self.d = kwargs.get('d')

    def run(self):
        pool = Pool(processes=4)
        results = []
        for i in range(10):
            res = pool.apply_async(self.func, (i, ))
            results.append(res)
            time.sleep(0.5)

        print('start...')

        for i, res in enumerate(results):
            print(i)
            results[i].get()

    def func(self, param):
        print('sleep for 5s...')
        time.sleep(5)
        print(self.a, self.b, self.c, self.d, param)
        raise ValueError

def func(a,b,c,d):
    print('sleep for 5s...')
    time.sleep(5)
    return a+b+c+d

def mapFunc(a):
    print('sleep for 5s...')
    time.sleep(5)
    return sum(a)

if __name__ == '__main__':
    pool = Pool(processes=4)
    results = []
    # for i in range(10):
    #     res = pool.apply_async(func, (1,2,3,i))
    #     res = pool.apply_async(func, kwds=con)
    #     tmp = test_class(**con)
    #     res = pool.apply_async(tmp.run)
    #     results.append(res)
    # t = test_class(**con)
    # t.run()
    # print('hey...')
    #
    # for i in range(10):
    #     print(i)
    #     results[i].get()

    params = list(map(lambda x:(1,2,3,x), list(range(100))))
    res = pool.map_async(mapFunc, params)

    print('hey...')

    print(res.get())