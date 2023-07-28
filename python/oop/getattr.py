class Test():
    def __init__(self):
        self.test = 100
    def forward(self, x):
        self.test = x


class Test2(Test):
    def __init__(self):
        super().__init__()
    def forwar(self, x):
        self.test = 10*x

test = Test2()

print(isinstance(test, Test))
if getattr(test, 'forward'):
    print('asd')