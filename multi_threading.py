import time
import threading


def calc_square(numbers):
    print('Calculating square numbers')
    for number in numbers:
        time.sleep(0.2)
        print('square', number * number)


def calc_cube(numbers):
    print('Calculate cube of numbers')
    for number in numbers:
        time.sleep(0.2)
        print('cube', number * number * number)


def main():
    arr = [2, 3, 8, 9]
    start_time = time.time()
    t1 = threading.Thread(target=calc_square, args=(arr,))
    t2 = threading.Thread(target=calc_cube, args=(arr,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    print("done in: {}".format(time.time() - start_time))
    print('Done with everything')


if __name__ == "__main__":
    main()
