import numc as nc


print("----------------------------")
print("testing slice")
print("----------------------------")

A = nc.Matrix(4, 3, 1)
print(A[0:3, 1:3])
print(A[1:3, 2])

B = nc.Matrix(3, 1, 2)
print(B[1:2])
print(B[0:3])

C = nc.Matrix(5, 1, 3)
print(C[1, 3])

