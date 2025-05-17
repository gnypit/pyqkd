import pickle
from material_endurance_test import fitness_function_pyqkd
from genetic_algorithm import Member

m = Member(
    genome=[0.5, 0.5, 0.5],
    identification_number=1,
    fitness_function=fitness_function_pyqkd
)

try:
    pickle.dumps(m)
    print("✅ Member instance is picklable")
except Exception as e:
    print(f"❌ Failed to pickle Member instance: {e}")
