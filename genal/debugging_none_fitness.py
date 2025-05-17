import multiprocessing
from material_endurance_test import fitness_function_pyqkd
import numpy as np

def worker(x):
    try:
        result = fitness_function_pyqkd(x)
        return result
    except Exception as e:
        print(f"Error in worker: {e}")
        return None

if __name__ == '__main__':
    with multiprocessing.Pool(1) as pool:
        xs = [
            [np.float64(0.7310373103731037), np.float64(0.4134641346413464), np.float64(0.2151821518215182),
             np.float64(0.7278772787727877), np.float64(0.16662166621666216), np.float64(0.9866498664986649)]
            ]
        results = pool.map(worker, xs)
        print(results)
