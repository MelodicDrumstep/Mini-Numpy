# numc

这个项目来自UCB CS61C。 在这个项目中， 我需要自己实现一个类似 numpy 的库， 并进行加速。

### 实现矩阵基本操作

这个没啥好说的， 就用 C 来写。 注意矩阵乘法顺序 ikj， 然后存临时变量。 矩阵乘法分块 / 循环展开 / AVX 向量化之类的后面再优化， 先把东西搞出来。

### 实现 python 与 C 的衔接

我一直很好奇为什么 python 可以“用 C 实现”。 这里给出了答案： 我可以先写个 C 库函数， 然后用 `<Python.h>` 里的一些规则实现和 python 里面的类和函数的映射。然后我直接把它编译， 编译完成后就是一个共享库， python 解释器遇到对应函数了就来调用这个共享库。

所以实际上， import 的库一般都不会是 python 写的。 然后除了这里的 `<Python.h>` 可以实现 python 接口、 C 底层之外， 还有更新的 Pybind11 之类的工具可以做， 而且 Pybind11 应该更好用。 我先把这里的写了， CMU DL SYS 那个课好像就有用 Pybind11 实现一些 python 库的 lab。 等这个写完了就去写那个。

这里的 `<Python.h>` 是这么做的：

首先， 我要有个 setup.py 用于建立这个库。(会编译我的 C 文件成 .so 文件)

```python
def main():
    CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
    LDFLAGS = ['-fopenmp']
    # 上面的就是些编译参数
    module = Extension('numc', sources = ['numc.c', 'matrix.c'], extra_compile_args = CFLAGS, extra_link_args = LDFLAGS)
    # 这是定义这个模块
    setup(name = 'numc', version = '1.0', description = 'numc', ext_modules = [module])
    # setup 这个模块， 调用编译器去编译
if __name__ == "__main__":
    main()
```

之后呢， 我还要实现好 numc.c。 我已经封装好了 matrix.c， 所以 numc.c 建立在 matrix 这一层之上。 接下来就有一堆奇怪的语法了：

```c numc.h
#include "matrix.h"
/*
 * Defines the struct that represents the object
 * Has the default PyObject_HEAD so it can be a python object
 * It also has the matrix that is being wrapped
 * is of type PyObject
 */
// This struct contains the actual matrix object pointer
// and PyObject_HEAD : to enable this struct be a python object
// and PyObject * shape : to store the shape of the matrix
typedef struct {
    PyObject_HEAD
    matrix* mat;
    PyObject *shape;
} Matrix61c;
```

一般源文件代码中注释我都写英文， 为了和国际接轨（）

然后它的函数也都是奇奇怪怪的形式。 需要传 `PyObject`, 类似 python 的类的函数格式。

```c numc.h
/* Function definitions */
static int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low, double high);
static int init_fill(PyObject *self, int rows, int cols, double val);
static int init_1d(PyObject *self, int rows, int cols, PyObject *lst);
static int init_2d(PyObject *self, PyObject *lst);
static void Matrix61c_dealloc(Matrix61c *self);
```

那具体是哪个 python 函数映射到哪个 c 函数呢？ 这个规则写在 `PyTypeObject` 里面：

```c
/* INSTANCE ATTRIBUTES*/
static PyMemberDef Matrix61c_members[] = {
    {"shape", T_OBJECT_EX, offsetof(Matrix61c, shape), 0,
     "(rows, cols)"},
    {NULL}  /* Sentinel */
};

static PyTypeObject Matrix61cType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numc.Matrix",
    .tp_basicsize = sizeof(Matrix61c),
    .tp_dealloc = (destructor)Matrix61c_dealloc,
    .tp_repr = (reprfunc)Matrix61c_repr,
    .tp_as_number = &Matrix61c_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,
    .tp_doc = "numc.Matrix objects",
    .tp_methods = Matrix61c_methods,
    .tp_members = Matrix61c_members,
    .tp_as_mapping = &Matrix61c_mapping,
    .tp_init = (initproc)Matrix61c_init,
    .tp_new = Matrix61c_new
};
```