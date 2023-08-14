class TreeSitterExample:

    # Class variable definition
    my_variable = 10

    def __init__(self):
        # Constructor code
        self.instance_variable = 20

    def my_method(self, parameter):
        # Method code
        self.my_variable = parameter

        # If Statement
        if parameter > 5:
            print("Parameter is greater than 5")

        # For Loop
        for i in range(10):
            print(f"For loop iteration: {i}")

        # While Loop
        while parameter < 10:
            print(f"While loop, parameter: {parameter}")
            parameter += 1

        # Try-Except-Finally
        try:
            if parameter == 8:
                raise ValueError("Random exception")
        except ValueError as e:
            print(f"Caught exception: {e}")
        finally:
            print("Finally block executed")

    # Lambda function
    addition = lambda self, a, b: a + b

    # Inner class definition
    class InnerClass:
        def inner_method(self):
            print("This is an inner class method")

    # Static method
    @staticmethod
    def static_example_method():
        print("Static method executed")

    # Class method
    @classmethod
    def class_example_method(cls):
        print("Class method executed")

