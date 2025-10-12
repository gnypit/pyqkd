from multiprocessing.managers import SyncManager, BaseManager
from multiprocessing import Value


class SimpleObject:
    a: float = None
    b: float = None

    def __init__(self, b):
        self.b = b

    def compute(self):
        self.a = self.b * 2
        return self.a


class AdvancedObject(SimpleObject):
    c: float = None

    def __init__(self, b):
        super().__init__(b)


class MyBaseManager(BaseManager):
    pass


class MySyncManager(SyncManager):
    pass


MyBaseManager.register('SimpleObject', SimpleObject)
MySyncManager.register('SimpleObject', SimpleObject)


class NormalObject:
    def __init__(self, b):
        self.b = b

    def compute(self):
        self.a = self.b * 2
        return self.a


if __name__ == '__main__':
    test0 = AdvancedObject(7)
    a0 = test0.compute()
    print(test0.a, a0)

    with MyBaseManager() as manager1:
        test1 = manager1.SimpleObject(7)
        a1 = test1.compute()
        print(a1)
        print(test1.a)

    with MySyncManager() as manager2:
        test2 = manager2.SimpleObject(7)
        a2 = test2.compute()
        print(a2)
        print(test2.a)

    # TODO: https://stackoverflow.com/questions/72708828/how-to-create-the-attribute-of-a-class-object-instance-on-multiprocessing-in-pyt
