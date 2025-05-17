from numpy import sum
from multiprocessing.sharedctypes import Value, Array
from multiprocessing import Process, Manager, Lock
from multiprocessing.managers import BaseManager
from ctypes import c_double

class SimpleChromosome:
    genes: list
    fit_val: float = None

    def __init__(self, genes):
        self.genes = genes

    def evaluate(self):
        self.fit_val = sum(self.genes)

class SharedChromosome:
    genes: Array
    fit_val: Value(c_double) = None

    def __init__(self, genes):
        self.genes = genes

    def evaluate(self):
        self.fit_val = sum(self.genes)

class ManagerChromosome:
    genes: list
    fit_val: Value(c_double) = None

    def __init__(self, genes):
        self.genes = genes
        self.fit_val = Value('d', 0.0)

    def evaluate(self):
        self.fit_val = float(sum(self.genes))


def evaluate_simple(chromosome: SimpleChromosome):
    chromosome.evaluate()
    print(f"Inside the process {chromosome.fit_val}")

def evaluate_shared(chromosome: SharedChromosome):
    chromosome.evaluate()
    print(f"Inside the process {chromosome.fit_val}")

def evaluate_manager(chromosome: ManagerChromosome):
    chromosome.evaluate()
    print(f"Inside the process {chromosome.fit_val}")

class ChromosomeManager(BaseManager):
    pass

ChromosomeManager.register('Chromosome', SimpleChromosome)

if __name__ == '__main__':
    simple = SimpleChromosome(genes=[1, 2, 3, 4])

    p = Process(target=evaluate_simple, args=(simple,))
    p.start()
    p.join()

    print(f"Outside the process {simple.fit_val}")

    shared = SharedChromosome(genes=[1, 2, 3, 4])
    p = Process(target=evaluate_shared, args=(shared,))
    p.start()
    p.join()

    print(f"Outside the process {shared.fit_val}")

    with Manager() as m:
        m_chrom = ManagerChromosome(genes=[1, 2, 3, 4])
        p = Process(target=evaluate_manager, args=(m_chrom,))
        p.start()
        p.join()

        print(f"Outside the process {m_chrom.fit_val}")

    with ChromosomeManager() as cm:
        cm_chrom = cm.Chromosome(genes=[1, 2, 3, 4])
        p = Process(target=evaluate_simple, args=(cm_chrom,))
        p.start()
        p.join()

        print(f"Outside the process {cm_chrom.fit_val}")
