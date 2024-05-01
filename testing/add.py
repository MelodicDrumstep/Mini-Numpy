from numc import Matrix

print("----------------------------")
print("testing add")
print("----------------------------")

a = Matrix(4, 2, 1)
b = Matrix(4, 2, 2)


print("下面这行应该成功")
print(a + b)
print("成功！")

a = Matrix(2, 3)
b = Matrix(3, 2)
c = Matrix(1, 2, 3)
d = Matrix(1, 2, 4)

print("下面这行应该成功")
print(c + d)
print("成功！")
print("下面这行应该报错")
print(a + b)

