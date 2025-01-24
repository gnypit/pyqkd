"""Inspired by James Cutajar: https://github.com/cutajarj/multithreadinginpython
Created for an online computational intelligence course with Helion.pl sp. z o.o."""
from multiprocessing import Process, Array, cpu_count
from time import time
from random import randint
from numpy import dot

"""To test the code, we create two 3x3 matrices:"""
matrix_a = [
    [3, 1, -4],
    [2, -3, 1],
    [5, -2, 0]
]

matrix_b = [
    [1, -2, -1],
    [0, 5, 4],
    [-1, -2, 3]
]

"""For a more robust test, 10x10 matrices:"""
matrix_big_a = [
    [-1, -3, 3, -5, 2, 4, 1, 2, -4, -4],
    [-5, -3, 1, -1, 0, 5, 2, -1, -4, -2],
    [0, -2, -5, -3, -2, -4, 5, 0, -4, -5],
    [-4, 2, 1, 1, 3, -4, 0, 2, 2, 0],
    [1, -3, -5, 4, -4, 2, -3, -5, 4, -1],
    [-4, 2, 0, -3, -5, -2, 2, 3, -2, -5],
    [3, 5, 2, -5, 5, -5, 3, -5, -3, -2],
    [3, 3, -3, -2, -2, -4, 2, -1, 0, -2],
    [-4, -5, -5, 4, -3, -4, 5, 3, -3, 3],
    [-4, -3, -5, -3, -4, 5, -4, 0, -2, 0]
]
matrix_big_b = [
    [5, 2, 1, -2, -5, -1, 2, -1, 0, 2],
    [4, 4, 4, -3, 1, -2, -2, 4, -3, -4],
    [1, 5, 1, 1, -1, 0, 4, -1, 0, -4],
    [-1, 2, -5, 5, -4, 2, -4, -5, 1, 5],
    [-1, 2, 1, -4, -4, 3, -5, -1, 5, -4],
    [3, 1, -3, 4, -2, 0, -2, -3, 0, -4],
    [-1, -4, 2, -1, -1, -4, -3, 5, -2, 2],
    [0, 2, 0, 0, 5, 2, 0, -2, 3, 2],
    [3, -5, 5, 2, 2, 0, 2, 0, -4, -4],
    [-1, -1, 0, 0, 4, -3, 3, 5, -2, 3]
]


def fill_in_matrix(matrix, size):
    for row in range(size):
        for col in range(size):
            matrix[row * size + col] = randint(-5, 5)


def matrix_multiplication(matrix1, matrix2, size=3):
    """Function takes matrices as 2D tables and multiplies them with nested loops."""
    result = [[0] * size for _ in range(size)]
    time_start = time()
    for row in range(size):
        for col in range(size):
            for i in range(size):
                result[row][col] += matrix1[row][i] * matrix2[i][col]
    time_end = time()
    t = time_end - time_start

    return result, t


def matrix_as_list_multiplication(matrix1, matrix2, size=3):
    """Function rewrites matrices from 2D tables into 1D tables and multiples them in this form."""
    m1 = [matrix1[i][j] for i, j in [[_i, _j] for _i in range(size) for _j in range(size)]]
    m2 = [matrix2[i][j] for i, j in [[_i, _j] for _i in range(size) for _j in range(size)]]
    result = [0] * (size ** 2)  # zakładamy, że nasze wszystkie macierze są kwadratowe

    time_start = time()
    for row in range(size):
        for col in range(size):
            for i in range(size):
                """From row and column number we create a single index."""
                result[row * size + col] += m1[row * size + i] * m2[i * size + col]
    end_time = time()
    t = end_time - time_start

    result_2d = [[0] * size for _ in range(size)]
    for index in range(len(result)):
        """From a single index we create again row and column number."""
        row = index // size
        col = index % size
        result_2d[row][col] = result[index]

    return result_2d, t


def compute_row(id, m1, m2, result, size, no_processes):
    """Tish function computes some rows of the matrix that will be the result of multiplying m1 & m2.
    If we have 10 processes and 200x200 matrices, and the id passed to this function is, e.g., 7,
    then this function will compute rows number: 7, 17, 27, 37, etc. (one at a time!).

   For this purpose, each time we take a single row from m1 and compute inner products with all the columns
   from matrix m2."""
    rows_to_compute = range(id, size, no_processes)

    """After identifying rows to compute (each process has different ones!) loops work similarly to the
    matrix_as_list_multiplication function:"""
    for row in rows_to_compute:
        for col in range(size):
            for i in range(size):
                """From a single index we create again row and column number."""
                result[row * size + col] += m1[row * size + i] * m2[i * size + col]


def test(m1, m2, size=3, print_result=True):
    numpy_time_start = time()
    result1 = dot(m1, m2)
    numpy_time_end = time()
    time1 = round(numpy_time_end - numpy_time_start, 3)

    result2, time2 = matrix_multiplication(m1, m2, size=size)
    result3, time3 = matrix_as_list_multiplication(m1, m2, size=size)

    time2, time3 = round(time2, 3), round(time3, 3)

    if print_result:
        print(f"Numpy returned {result1} in {time1} seconds\n"
              f"First func returned {result2} in {time2} seconds\n"
              f"Second func returned {result3} in {time3} seconds")
    else:
        print(f"Numpy returned result in {time1} seconds\n"
              f"First func returned result in {time2} seconds\n"
              f"Second func returned result in {time3} seconds")


if __name__ == '__main__':
    test(m1=matrix_big_a, m2=matrix_big_b, size=10)  # test(m1=matrix_a, m2=matrix_b)

    """We initiate matrices as objects in the shared memory"""
    big_a = [matrix_big_a[i][j] for i, j in [[_i, _j] for _i in range(10) for _j in range(10)]]
    big_b = [matrix_big_b[i][j] for i, j in [[_i, _j] for _i in range(10) for _j in range(10)]]

    matrix1_array = Array('i', big_a, lock=False)  # we don't need the blockade, processes only read this Array
    matrix2_array = Array('i', big_b, lock=False)  # we don't need the blockade, processes only read this Array
    result_array = Array('i', [0] * 100, lock=False)  # we don't need the blockade, processes have access to separate parts for modification

    """We create and start the processes and time measurement."""
    process_count = 5
    processes = []
    process_time_start = time()
    for proc_id in range(process_count):
        p = Process(target=compute_row, args=(
            proc_id, matrix1_array, matrix2_array, result_array, 10, process_count
        ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Processes worked for {time() - process_time_start:.2f}s")
    print(result_array[:])

    """Now for slightly bigger matrices"""
    matrix_size = 200
    matrix1_array = Array('i', [0] * (matrix_size ** 2), lock=False)
    matrix2_array = Array('i', [0] * (matrix_size ** 2), lock=False)
    result_array = Array('i', [0] * (matrix_size ** 2), lock=False)

    fill_in_matrix(matrix=matrix1_array, size=matrix_size)
    fill_in_matrix(matrix=matrix2_array, size=matrix_size)

    process_count = cpu_count()
    processes = []
    process_time_start = time()
    for proc_id in range(process_count):
        p = Process(target=compute_row, args=(
            proc_id, matrix1_array, matrix2_array, result_array, matrix_size, process_count
        ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    process_time_end = time()

    result_array = Array('i', [0] * (matrix_size ** 2), lock=False)
    sequential_time_start = time()
    for row in range(matrix_size):
        for col in range(matrix_size):
            for i in range(matrix_size):
                result_array[row * matrix_size + col] += matrix1_array[row * matrix_size + i] * matrix2_array[
                    i * matrix_size + col
                    ]
    sequential_time_end = time()

    print(f"Processes worked for {process_time_end - process_time_start} seconds\n"
          f"Sequential code worked for {sequential_time_end - sequential_time_start} seconds")
