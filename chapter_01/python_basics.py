"""
Run with: python python_basics.py
"""

# Simple values and types
name = "Alice"
age = 30
pi = 3.14159
is_active = True

# Print and formatting
def demo_printing():
    print("--- Printing & Formatting ---")
    print(f"Name: {name}, Age: {age}")
    print("PI rounded:", round(pi, 2))

# Basic arithmetic and assignment
def demo_arithmetic():
    print("--- Arithmetic ---")
    a = 10
    b = 3
    print(a + b, a - b, a * b, a / b)
    print("Integer division:", a // b, "mod:", a % b)

# Strings
def demo_strings():
    print("--- Strings ---")
    s = "Hello, World!"
    print(s.upper(), s.lower(), s.replace("World", "Python"))

# Lists and list operations
def demo_lists():
    print("--- Lists ---")
    fruits = ["apple", "banana", "cherry"]
    fruits.append("date")
    print("Fruits:", fruits)
    print("Looping:")
    for i, f in enumerate(fruits, 1):
        print(i, f)

# Dictionaries
def demo_dicts():
    print("--- Dictionaries ---")
    person = {"name": "Bob", "age": 25}
    person["city"] = "New York"
    print(person)
    print("Keys:", list(person.keys()))

# Conditionals and loops
def demo_control_flow(n=5):
    print("--- Control Flow ---")
    if n > 0:
        print(n, "is positive")
    else:
        print(n, "is not positive")

    print("While loop up to", n)
    i = 0
    while i < n:
        print(i, end=" ")
        i += 1
    print()

# Functions and return values
def add(x, y):
    return x + y

def demo_functions():
    print("--- Functions ---")
    print("add(2,3)=", add(2, 3))

# Exception handling example
def demo_exceptions():
    print("--- Exceptions ---")
    try:
        x = int("not an int")
    except ValueError as e:
        print("Caught ValueError:", e)

# File I/O (writes and reads a temporary file)
def demo_file_io():
    print("--- File I/O ---")
    fname = "sample.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("This is a sample file.\nLine 2.\n")
    print(f"Wrote to {fname}")
    with open(fname, "r", encoding="utf-8") as f:
        print("Contents:")
        print(f.read())

# Small demo runner
def main():
    demo_printing()
    demo_arithmetic()
    demo_strings()
    demo_lists()
    demo_dicts()
    demo_control_flow(3)
    demo_functions()
    demo_exceptions()
    demo_file_io()
    print("\nTutorial demo complete.")

if __name__ == "__main__":
    main()
