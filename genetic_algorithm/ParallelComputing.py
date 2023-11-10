from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
from threading import Thread
from time import perf_counter


# Computables

class Computable():
    """Base Computable class"""

    def __init__(self, **kwargs) -> None:
        self.arguments = kwargs

    def compute(self):
        print(f'--> computing {self.__class__.__name__} with {self.arguments}')


class ComputeSum(Computable):

    def compute(self):
        super().compute()

        assert 'a' in self.arguments
        assert 'b' in self.arguments
        assert 'c' in self.arguments

        sum = self.arguments['a'] + self.arguments['b'] + self.arguments['c']
        return f'Simple sum: {sum}'


class ComputePowerManyTimes(Computable):

    def compute(self):
        super().compute()

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

class BaseExecutor:
    """Base Excecutor class"""

    threads = []

    def __init__(self, computables: list[Computable]) -> None:
        """Pass here the list of objects imlementing the Computable"""

        self.computables = computables

    def execute(self) -> None:
        pass

    def run(self):
        print(f'\n\n----> Running {self.__class__.__name__}')
        start_time = perf_counter()
        self.execute()
        for t in self.threads:
            t.join()
        end_time = perf_counter()
        print(f'It took {end_time - start_time: 0.2f} second(s) to complete.')


class SerialExecutor(BaseExecutor):
    """This class computes the Computables in a simple loop"""

    def execute(self):
        for computable in self.computables:
            print(computable.compute())


class ProcessPoolParallelExecutor(BaseExecutor):
    """
    This class uses the ProcessPoolExecutor to compute Computables in parralel
    To learn more about ProcessPoolExecutor see i.e. https://superfastpython.com/threadpoolexecutor-in-python/
    """

    def exec_single(self, computable):
        return computable.compute()

    def execute(self):
        executor = ProcessPoolExecutor()

        # use map to run `exec_single` for each of `computables`
        results = executor.map(self.exec_single, self.computables)

        for result in results:
            print(result)


class ThreadParallelExecutor(BaseExecutor):

    def execute(self):
        for computable in self.computables:
            th = Thread(target=computable.compute)
            th.start()
            self.threads.append(th)


class ProcessParallelExecutor(BaseExecutor):

    def execute(self):
        for computable in self.computables:
            pr = Process(target=computable.compute)
            pr.start()
            self.threads.append(pr)


if __name__ == '__main__':
    computables_list = []

    # building the sample list of Computable objects
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=20000000))
    computables_list.append(ComputeSum(a=1, b=2, c=3))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=20000000))
    computables_list.append(ComputeSum(a=4, b=5, c=6))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=20000000))
    computables_list.append(ComputeSum(a=7, b=8, c=9))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=20000000))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=20000000))
    computables_list.append(ComputePowerManyTimes(base=5, exp=10, iterations=20000000))

    serial_executor = SerialExecutor(computables_list)
    serial_executor.run()

    parallel_executor = ProcessPoolParallelExecutor(computables_list)
    parallel_executor.run()

    parallel_executor = ThreadParallelExecutor(computables_list)
    parallel_executor.run()

    parallel_executor = ProcessParallelExecutor(computables_list)
    parallel_executor.run()
