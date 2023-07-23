import inspect

def a():
    for i in range(10):
        yield i

print(inspect.isgeneratorfunction(a))
