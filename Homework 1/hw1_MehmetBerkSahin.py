
class Sequence(object):
    """
    A class to represent a sequence.

    Attributes
    ----------
    array : list
        sequence as a list

    Methods
    -------
    __gt__(other):
        Performs element-wise > between arrays. And returns the number of true values.
    __iter__():
        Creates a iterator for Fibonacci object.
    __len__():
        Print the length of the Fibonacci sequence.
    """

    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        return SeqIterable(self)

    def __gt__(self, other):
        """
        Performs element-wise > operation on the elements of arrays. Returns
        the number of True values.

        Parameters
        ----------
        other : Sequence, required
            another Sequence object for comparison

        Returns
        -------
        Number of True values after the comparison.
        """

        counter = 0
        if len(self.array) != len(other):
            raise ValueError("Two arrays are not equal in length!")
        else:
            for idx in range(len(self.array)):
                if self.array[idx] > other.array[idx]:
                    counter += 1 # count the true values
        return counter

class SeqIterable(object):
    """
    Iterator class for Sequence class.

    Attributes
    ----------
    items : list
        sequence of prime numbers as a list
    index : int
        index that iterator follows in the sequence

    Methods
    -------
    __iter__():
        Returns an iterator
    __next__():
        Returns the element in the sequence at the current index

    """

    def __init__(self, seq_obj):
        self.items = seq_obj.array
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1 # initialize index as 0
        # make sure index is in the range of the sequence
        if self.index < len(self.items):
            return self.items[self.index]
        else:
            raise StopIteration

class Fibonacci(Sequence):
    """
    A class to represent Fibonacci sequences.

    Attributes
    ----------
    first_value : int
        first number of the Fibonacci sequence
    second_value : int
        second number of the Fibonacci sequence
    array : list
        sequence as a list

    Methods
    -------
    __call__(length):
        Creates a Fibonacci sequence with the given length and print it.
    """

    def __init__(self, first_value, second_value):
        super(Fibonacci, self).__init__([])
        self.first_value = first_value
        self.second_value = second_value

    def __call__(self, length):
        fib_seq = []
        for i in range(length):
            if i == 0:
                fib_seq.append(self.first_value)
            elif i == 1:
                fib_seq.append(self.second_value)
            else:
                new_element = fib_seq[-1] + fib_seq[-2] # Fibonacci rule
                fib_seq.append(new_element)
        # Save the result and print it
        self.array = fib_seq
        print(self.array)

class Prime(Sequence):
    """
    A class to represent sequence of prime numbers.

    Attributes
    ----------
    array : list
        sequence of prime numbers as a list

    Methods
    -------
    __call__(length):
        Generates a sequence with the given length and consists of prime numbers.
    """

    def __init__(self):
        super(Prime, self).__init__([])

    def __call__(self, length):
        prime_seq = [] # sequence of prime numbers
        if length >= 1:
            prime_seq.append(2) # smallest prime number
        for i in range(length-1):
            new_num = prime_seq[-1] # continue for the prime number search from the last
            primeFound = False # flag for whether prime number is found
            while not primeFound:
                primeFound = True
                new_num += 1
                for prime in prime_seq:
                    # prime is found if the below statement is never true
                    if new_num % prime == 0:
                        primeFound = False
                        break
            # out of while means we found prime and we can add it to seq.
            prime_seq.append(new_num)
        self.array = prime_seq # save the prime sequence
        print(self.array)

# ask user to either reproduce the results in the snippets or
# produce results with the parameters of my choice
choice = int(input("Reproduce the results in the snippet (enter 0), run author's inputs (enter 1): "))

while not ((choice == 1) or (choice == 0)):
    print("Your answer is invalid. Enter a valid answer.")
    choice = int(input("Reproduce the results in the snippet (enter 0), run author's inputs (enter 1): "))
# reproduce the results in the code snippets of the assignment
if choice == 0:
    print("Results with the parameters given in the homework.")
    # Q3
    print("-"*10 + "Result of question 3" + "-"*10)
    FS = Fibonacci(first_value=1, second_value=2)
    FS(length=5)
    print("-"*40)

    # Test for Fibonacci Sequence (Q4)
    print("-" * 10 + "Result of question 4" + "-" * 10)
    FS = Fibonacci(first_value=1, second_value=2)
    FS(length=5)
    print(len(FS))
    print([n for n in FS])
    print("-"*40)

    # Test for Prime Sequence (Q5)
    print("-" * 10 + "Result of question 5" + "-" * 10)
    PS = Prime()
    PS(length=8)
    print(len(PS))
    print([n for n in PS])
    print("-"*40)

    # Test for Comparison Operator (Q6)
    print("-" * 10 + "Result of question 6" + "-" * 10)
    FS = Fibonacci(first_value=1, second_value=2)
    FS(length=8)
    PS = Prime()
    PS(length=8)
    print(FS > PS)
    PS(length=5)
    print(FS > PS)
    print("-"*40)
else:
    # produce the results with the parameters of my choice
    print("Results with the parameters of my choice. ")
    # Q3
    print("-" * 10 + "Result of question 3" + "-" * 10)
    FS = Fibonacci(first_value=1, second_value=5)
    FS(length=7)
    print("-" * 40)

    # Test for Fibonacci Sequence (Q4)
    print("-" * 10 + "Result of question 4" + "-" * 10)
    FS = Fibonacci(first_value=1, second_value=1)
    FS(length=8)
    print(len(FS))
    print([n for n in FS])
    print("-" * 40)

    # Test for Prime Sequence (Q5)
    print("-" * 10 + "Result of question 5" + "-" * 10)
    PS = Prime()
    PS(length=10)
    print(len(PS))
    print([n for n in PS])
    print("-" * 40)

    # Test for Comparison Operator (Q6)
    print("-" * 10 + "Result of question 6" + "-" * 10)
    FS = Fibonacci(first_value=0, second_value=2)
    FS(length=4)
    PS = Prime()
    PS(length=4)
    print(FS > PS)
    PS(length=0)
    print(FS > PS)
    print("-" * 40)
