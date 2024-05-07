from asyncio import sleep
import asyncio
import ctypes
from multiprocessing import Array, Process, RawArray
from time import perf_counter


# Computables

class Computable():

    def __init__(self, **kwargs) -> None:
        self.arguments = kwargs

    def print_arguments(self) -> str:
        for key, value in self.arguments:
            print(f'{key}: {value}')

    def compute(self):
        pass


class ComputeSum(Computable):

    def compute(self):
        assert 'a' in self.arguments
        assert 'b' in self.arguments
        assert 'c' in self.arguments

        sum = self.arguments['a'] + self.arguments['b'] + self.arguments['c']
        return f'Simple sum: {sum}'


class ComputePowerManyTimes(Computable):

    def compute(self):
        assert 'base' in self.arguments
        base = self.arguments['base']

        assert 'exp' in self.arguments
        exp = self.arguments['exp']

        assert 'iterations' in self.arguments
        iterations = self.arguments['iterations']

        sum = 0
        for i in range(0, iterations):
            sum += base ** exp

        return f'Complex sum: {sum}'


# Executors

class BaseExecutor():
    def __init__(self, computables: [Computable]) -> None:
        print(f'Creating {self.__class__.__name__}')
        self.computables = computables

    def execute(self) -> None:
        pass


class SerialExecutor(BaseExecutor):

    def execute(self):
        for computable in self.computables:
            print(computable.compute())


class ParallelExecutor(BaseExecutor):
    # TODO: use ThreadPoolExecutor

    processes = []

    def _exec_one(self, computable, counter, results):
        """Wrapper function for calling one Computable compute() and storing the result        

        Args:
            computable (Computable): Computable object to call the compute() on
            counter (int): sequence number of given Computable
            results (RawArray): Shared array to store the Computable result
        """
        res = computable.compute()
        # need to convert string to bytes for this array
        results[counter].value = str.encode(res)

    def execute(self):
        # prepare shared array for storing the computable results
        self.results = [RawArray(ctypes.c_char, 100) for _ in range(len(self.computables))]

        for counter in range(len(self.computables)):
            pr = Process(target=self._exec_one,
                         args=(self.computables[counter], counter, self.results))
            self.processes.append(pr)
            pr.start()

    def get_num_active_threads(self) -> int:
        sum = 0
        for th in self.processes:
            if th.is_alive():
                sum += 1
        return sum


if __name__ == '__main__':
    print('hello')

    computables_list = []

    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=2000000))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=3000000))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=1000000))
    computables_list.append(ComputeSum(a=1, b=2, c=3))
    computables_list.append(ComputeSum(a=4, b=5, c=6))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=5000000))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=5000000))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=5000000))
    computables_list.append(ComputeSum(a=7, b=8, c=9))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=5000000))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=5000000))

    print('\nSerial execution:')
    start_time = perf_counter()
    serial_executor = SerialExecutor(computables_list)
    serial_executor.execute()
    end_time = perf_counter()
    print(f'It took {end_time - start_time: 0.2f} second(s) to complete.')

    print('\nParallel execution:')
    start_time = perf_counter()
    parallel_executor = ParallelExecutor(computables_list)
    parallel_executor.execute()

    # wait for the threads to complete
    # for p in parallel_executor.processes:
    #    p.join()    

    # you can do other stuff here!
    print('\n...doing other stuff in the main thread while there are ' +
          f'{parallel_executor.get_num_active_threads()} active Computable processess...')

    secs = 4
    while secs := secs - 1:
        print(f'countdown ({secs})...', end='\r', flush=True)
        # asyncio forces main thread to wait seconds while parallel processing happens
        asyncio.run(sleep(1))

    print('...still in the main thread while there are ' +
          f'{parallel_executor.get_num_active_threads()} active Computable processes...\n')

    # now wait in a loop for all Computable processes to finish
    while active := parallel_executor.get_num_active_threads():
        print(f'Active Computables: {active}  ', end='\r', flush=True)

    print('All Computable processes finished - results:')
    for result in parallel_executor.results:
        print(result.value.decode())
    end_time = perf_counter()

    print(f'It took {end_time - start_time: 0.2f} second(s) to complete.')
