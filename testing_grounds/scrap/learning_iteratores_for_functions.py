
from typing import Iterator


'''
def some_function(value):
    if value > 10:
        yield "greater than 10"
        value = value - 1
    else:
        yield "not greater than 10"


some_input = 20

my_generator = some_function( some_input )

for item in my_generator:
    print( item )


def fibonacci_generator(limit):
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b

# Using the generator
fib_gen = fibonacci_generator(10)
for num in fib_gen:
    print(num)
'''

money1 = 10
money2 = 7
money3 = 4
money4 = 1

def buy_priority(money) -> Iterator[str]:
    if money >= 8:
        yield "province"
    if money >= 6:
        yield "gold"
    if money >= 3:
        yield "silver"


results = buy_priority(money1)
#my_list = list(result)
#print( my_list )

#print( next( result ) )

for result in results:
    print( result )

'''
for r in result:
    print( result )
'''





''' This doesn't work.
def some_function(value: int) -> Iterator[str]:
    if value > 10:
        yield "greater than 10"
    else:
        yield "not greater than 10"


def print_some_function(some_function: Iterator[str]):
    for x in some_function:
        print(x)

some_input = [ 10, 0, 10, 10, 0, 0 ]

for x in some_input:
    print( some_function( x ) )


#print_some_function( some_function( some_input ) )
'''
