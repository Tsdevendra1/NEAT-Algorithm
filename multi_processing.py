import time
import multiprocessing


def calc_square(numbers, result):
    print('Calculating square numbers')
    for number in numbers:
        time.sleep(0.2)
        print('square', number * number)


def calc_cube(numbers, result, value, queue):
    print('Calculate cube of numbers')
    value.value = 5
    for index, number in enumerate(numbers):
        time.sleep(0.2)
        queue.put(number * number * number)
        result[index] = number * number * number


def main():
    # SHARED MEMORY CONCEPT
    arr = [2, 3, 8, 9]
    start_time = time.time()
    # We have to create a shared memory variable. Specify data type and size
    result = multiprocessing.Array('i', 4)
    value = multiprocessing.Value('i', 0)
    queue = multiprocessing.Queue()
    # p1 = multiprocessing.Process(target=calc_square, args=(arr,))
    p2 = multiprocessing.Process(target=calc_cube, args=(arr, result, value, queue))

    # p1.start()
    p2.start()

    # p1.join()
    p2.join()

    while not queue.empty():
        print('Getting')
        print(queue.get())
    print(result[:])
    print(value.value)
    print("done in: {}".format(time.time() - start_time))
    print('Done with everything')


import time
import multiprocessing


def deposit(balance, lock):
    for i in range(100):
        time.sleep(0.01)
        lock.acquire()
        balance.value = balance.value + 1
        lock.release()


def withdraw(balance, lock):
    for i in range(100):
        time.sleep(0.01)
        lock.acquire()
        balance.value = balance.value - 1
        lock.release()


def main2():
    # LOCK CONCEPT
    balance = multiprocessing.Value('i', 200)
    # Lock is used to lock a variable so that the value isn't frozen in time and when it is changed, it is not changed fast
    # enough for the other prcoess to know the updated value thus using the old value of the variable
    lock = multiprocessing.Lock()
    d = multiprocessing.Process(target=deposit, args=(balance, lock))
    w = multiprocessing.Process(target=withdraw, args=(balance, lock))
    d.start()
    w.start()
    d.join()
    w.join()
    print(balance.value)


from multiprocessing import Pool


def f(n):
    return n * n


def main3():
    # MAP AND REDUCE CONCEPT
    array = [1, 2, 3, 4, 5, 6]

    start_time = time.time()
    p = Pool()
    result = p.map(f, array)
    end_time = time.time() - start_time
    print(result, end_time)

    start_time = time.time()
    squared = []
    for n in array:
        time.sleep(5)
        squared.append(n * n)
    end_time = time.time() - start_time
    print(squared, end_time)


if __name__ == '__main__':
    # main()
    # main2()
    main3()
