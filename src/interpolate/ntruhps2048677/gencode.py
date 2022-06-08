#!/usr/bin/env python3
import sys

registered_func = set()

def operation(func):
    registered_func.add(func.__name__)
    return func

@operation
def basemul():
    for i in range(63):
        print(f'    schoolbook_KxK(r + {i} * 2 * K, a + {i} * K, b + {i} * K);')

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in registered_func:
        eval(f'{sys.argv[1]}()')
    else:
        print('usage: ./gencode.py [operation]')
