import numc as nc

print("----------------------------")
print("testing sub")
print("----------------------------")

A = nc.Matrix([[1, 2, 3], [4, 5, 6]])
B = nc.Matrix(2, 3, 3)
print(f'A - B = {A - B}')


