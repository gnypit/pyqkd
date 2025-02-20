"""Inspired by James Cutajar: https://github.com/cutajarj/multithreadinginpython
Created for an online computational intelligence course with Helion.pl sp. z o.o."""

#Parallel, delayed, randint sys, numpy
from time import time
from random import randint
import numpy as np
from joblib import Parallel, delayed
import sys

# Number cores
NUM_CORES=3

def numpy_dot(m1:list, m2:list):
    """
    Multiplica duas matrizes usando numpy.dot e mede o tempo de execução.

    Args:
        m1 (numpy.ndarray): first  matrix.
        m2 (numpy.ndarray): second matrix.

    Returns:
        tuple: (resultado da multiplicação, tempo de execução)
    """
    numpy_time_start = time()
    result1 = np.dot(m1, m2)
    numpy_time_end = time()

    return result1,numpy_time_start-numpy_time_end
    
def fill_in_matrix(size:int,maxint:int,minint:int):
    """
    Generates a square matrix of a given size filled with random integers within a specified range.

    Parameters:
        size (int): The size of the square matrix (number of rows and columns).
        maxint (int): The maximum integer value (inclusive) for the random numbers.
        minint (int): The minimum integer value (inclusive) for the random numbers.

    Returns:
        np.array: A square matrix of the specified size filled with random integers.
    """

    matrix=[]
    for _ in range(size):
        array=[]        
        for _ in range(size):
            array.append(randint(maxint,minint))
        matrix.append(array)
    return matrix

def multiplication_row_column(row:list[int],column:list[int]):
    """
    Computes the dot product of two vectors (lists or arrays).

    Parameters:
        vector1 (list or np.array): The first vector.
        vector2 (list or np.array): The second vector.

    Returns:
        int or float: The dot product of the two vectors.
    """

    result=0    
    for i in range(len(row)):
        result+=row[i]*column[i]
    return result
        
def matrix_multiplication(matrix1:np.array, matrix2:np.array):
    """
    Performs matrix multiplication using parallel processing.

    Parameters:
        matrix1 (np.array): The first matrix (m x n).
        matrix2 (np.array): The second matrix (n x p).

    Returns:
        np.array: The resulting matrix (m x p).
        float: The time taken for the computation.
    """

    matrix2_t=np.transpose(matrix2)
    time_start = time()
    result=Parallel(n_jobs=NUM_CORES)(delayed(multiplication_row_column)(i,j)for i in matrix1 for j in matrix2_t)
    time_end = time()
    t = time_end - time_start

    return result,t



def matrix_as_list_multiplication(matrix1, matrix2):
    """
    Multiplica duas matrizes transformando-as em listas 1D para otimizar a operação.
    
    Parameters:
        matrix1 (list of list): Primeira matriz (2D).
        matrix2 (list of list): Segunda matriz (2D).
    
    Returns:
        tuple: Matriz resultante (2D) e tempo de execução.
    """
    size = len(matrix1)


    m1 = [matrix1[i] for i in range(size)]
    m2_transposed = [[matrix2[j][i] for j in range(size)] for i in range(size)]  # Transposição

    time_start = time()

    result = Parallel(n_jobs=-1)(
        delayed(multiplication_row_column)(m1[i], m2_transposed[j])
        for i in range(size) for j in range(size)
    )

    end_time = time()
    elapsed_time = end_time - time_start

    result_2d = [result[i * size:(i + 1) * size] for i in range(size)]

    return result_2d, elapsed_time




def test(m1, m2,print_result=True):
    """
    Function to compare the performance of three different matrix multiplication methods:
        1. Using the `numpy.dot` function (numpy_dot).
        2. Using a custom matrix multiplication implementation (matrix_multiplication).
        3. Using a custom matrix multiplication implementation with matrices represented as lists (matrix_as_list_multiplication).

    Parameters:
        - m1: First matrix to be multiplied.
        - m2: Second matrix to be multiplied.
        - print_result: If True, prints the results of the multiplications. Otherwise, only prints the execution times.

    Returns:
        - The function does not return any value; it only prints the results and execution times.
    """

    result1, t_dot_numpy=numpy_dot(m1, m2)
    result2, t_dot_matrix_multiplication=matrix_multiplication(m1, m2)
    result3, t_dot_list_multiplication=matrix_as_list_multiplication(m1,m2)


    

    if print_result:
        print(f"Numpy returned {result1} in {t_dot_numpy:.3f} seconds\n")
        print(f"Numpy returned {result2} in {t_dot_matrix_multiplication:.3f} seconds\n")
        print(f"Numpy returned {result3} in {t_dot_matrix_multiplication:.3f} seconds\n")
    else:
        print(f"Numpy returned result in {t_dot_numpy:.3f} seconds\n")
        print(f"multiplication matrix returned result in {t_dot_matrix_multiplication:.3f} seconds\n")
        print(f"multiplication list matrix returned result in {t_dot_list_multiplication:.3f} seconds\n")



if __name__ == '__main__':
    
    matrix_big_a=fill_in_matrix(int(sys.argv[1]),-5,5)
    matrix_big_b=fill_in_matrix(int(sys.argv[1]),-5,5)

    test(m1=matrix_big_a, m2=matrix_big_b)


    

 