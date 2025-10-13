from multiprocessing import Value
from multiprocessing.managers import SyncManager, BaseManager


class SimpleObject:
    def __init__(self, b):
        self.b = Value('i', b)

    def compute(self):
        self.a = Value('i', self.b.value * 2)
        return self.a


class AdvancedObject(SimpleObject):
    def __init__(self, b):
        super().__init__(b)


class MyBaseManager(BaseManager):
    pass


class MySyncManager(SyncManager):
    pass


MyBaseManager.register('SimpleObject', SimpleObject)
MySyncManager.register('SimpleObject', SimpleObject)


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
