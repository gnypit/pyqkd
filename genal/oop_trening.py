class autko:
    def __init__(self):
        print("Powstałem")
    def dodaj_paliwo(self, ilosc):
        self.ilosc = ilosc
    def podaj_ilosc(self):
        print(self.ilosc)

bmw = autko()
bmw.dodaj_paliwo(10)
bmw.podaj_ilosc()

audi = autko()
audi.dodaj_paliwo(20)
audi.podaj_ilosc()

a = []
'''a.append()'''



class crew:
    def __init__(self, name, age, position):
        self.name = name
        self.age = age
        self.position = position


kirk = crew('James Kirk', 34, 'Captain')
spock = crew('Spock', 35, 'Officer')
mmcoy = crew('Leonard McCoy', 52, 'Medical Officer')

print(kirk.name, kirk.age, kirk.position)



'''Exercise: car class'''
class cars:
    def __init__(self, brand, color, mileage):
        self.brand = brand
        self.color = color
        self.mileage = mileage

    def description(self):
        return f"{self.brand} is {self.color} car with {self.mileage} miles"

bmw = cars('BMW', 'Black', 20000)
audi = cars('Audi', 'Red', 50000)
opel = cars('Opel', 'Blue', 10000)

for cars in (bmw, audi, opel):
    print(cars.description())


'''Exercise: Inheritance'''

class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name} is {self.age} years old"

    def speak(self, sound):
        return f"{self.name} says {sound}"

class GoldenRetriver(Dog):
    def speak(self, sound="Bark"):
        return super().speak(sound)


golden = GoldenRetriver('Hapsio', 4)
print(golden.speak())


