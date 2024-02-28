class DM:
    def __init__(self):
        pass

    def foo(self):
        print('DM_foo')

    def func(self):
        pass


class A_DM(DM):
    def __init__(self):
        super().__init__()
        self.txt = 'A_DM'

    def foo(self):
        print('ADM_foo')

    def func(self):
        print(self.txt)


class B_DM(DM):
    def __init__(self):
        super().__init__()
        self.txt = 'B_DM'

    def func(self):
        print(self.txt)


def func(self):
    print('patched_' + self.txt)

#setattr(A_DM, 'func', func)

a = A_DM()
a.func()
a.foo()
