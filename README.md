# numc

这个项目来自UCB CS61C。 在这个项目中， 我需要自己实现一个类似 numpy 的库， 并进行加速。

### 实现矩阵基本操作

这个没啥好说的， 就用 C 来写。 注意矩阵乘法顺序 ikj， 然后存临时变量。 矩阵乘法分块 / 循环展开 / AVX 向量化之类的后面再优化， 先把东西搞出来。

### 实现 python 与 C 的衔接

这部分要参考 https://docs.python.org/3.6/c-api/structures.html Python的 C 拓展的官方文档。

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
/* INSTANCE METHOD */
// 这个就是给我的 python 类增添一些 method
static PyMethodDef Matrix61c_methods[] = {
    /* TODO: YOUR CODE HERE */
    {NULL, NULL, 0, NULL}
};

/* INSTANCE ATTRIBUTES*/
// 这个就是给我的 python 类增添成员， 这里增添了一个 shape 成员
// 里面有些用法现用现查就行
static PyMemberDef Matrix61c_members[] = {
    {"shape", T_OBJECT_EX, offsetof(Matrix61c, shape), 0,
     "(rows, cols)"},
    {NULL}  /* Sentinel */
};

// 这个是写映射关系， 哪个 python 函数对应哪个 c 函数
static PyTypeObject Matrix61cType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numc.Matrix", //类型名称
    .tp_basicsize = sizeof(Matrix61c), //对象基础所占内存大小
    .tp_dealloc = (destructor)Matrix61c_dealloc, //析构函数
    .tp_repr = (reprfunc)Matrix61c_repr, // 对象的字符串表示函数
    .tp_as_number = &Matrix61c_as_number, // 指向数字方法的结构体
    .tp_flags = Py_TPFLAGS_DEFAULT | 
        Py_TPFLAGS_BASETYPE, // 类型标志
    .tp_doc = "numc.Matrix objects", //文档
    .tp_methods = Matrix61c_methods, // methods 是上方定义好的
    .tp_members = Matrix61c_members, // members 也是上方定义好的
    .tp_as_mapping = &Matrix61c_mapping, // 指向映射方法的结构体， 允许对象表现得像 dict
    .tp_init = (initproc)Matrix61c_init, //__init__函数， 是构造函数完成之后再调用的初始化函数， 主要是根据参数初始化对象的状态
    .tp_new = Matrix61c_new // 类型的构造函数， 主要是分配内存
};
```

然后在这里， 很不幸的是， 运行 test_correctness 立刻 segmentation fault...

也让我学会了如何调试 python 的 C 拓展， gdb 居然还可以调这个， gdb yyds

```shell
gdb --args python testing/test_correctness.py
```

另外， 原来运行 python 的 C 接口要多开这么多线程（？也许我理解有问题）

```shell
(gdb) run
Starting program: /mnt/d/Mini_Numpy/.venv/bin/python testing/test_correctness.py
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
CS61C Summer 2020 Project 4: numc imported!
[New Thread 0x7ffff43ff640 (LWP 1005841)]
[New Thread 0x7fffebbfe640 (LWP 1005842)]
[New Thread 0x7fffeb3fd640 (LWP 1005843)]
[New Thread 0x7fffdabfc640 (LWP 1005844)]
[New Thread 0x7fffda3fb640 (LWP 1005845)]
[New Thread 0x7fffd1bfa640 (LWP 1005846)]
[New Thread 0x7fffc13f9640 (LWP 1005847)]
[New Thread 0x7fffb8bf8640 (LWP 1005848)]
[New Thread 0x7fffb03f7640 (LWP 1005849)]
[New Thread 0x7fffa7bf6640 (LWP 1005850)]
[New Thread 0x7fff9f3f5640 (LWP 1005851)]
[New Thread 0x7fff9ebf4640 (LWP 1005852)]
[New Thread 0x7fff8e3f3640 (LWP 1005853)]
[New Thread 0x7fff85bf2640 (LWP 1005854)]
[New Thread 0x7fff853f1640 (LWP 1005855)]
[New Thread 0x7fff74bf0640 (LWP 1005856)]
[New Thread 0x7fff6c3ef640 (LWP 1005857)]
[New Thread 0x7fff63bee640 (LWP 1005858)]
[New Thread 0x7fff633ed640 (LWP 1005859)]

Thread 1 "python" hit Breakpoint 1, PyCFuncPtr_new (type=0x555555a713c8, args=0x7ffff514d7b8, kwds=0x0) at /mnt/d/Python-3.6.8/Modules/_ctypes/_ctypes.c:3557
3557        self->thunk = thunk;
```

我逐渐理解一切。。。 python 里的对象都是用指针访问的， 和 C++ 智能指针一样， 如果对象的 refcnt 减到 0 了就释放那块内存。（这应该算自动垃圾回收吧）

我又逐渐理解一切。。。原来每次改完 C 底层之后都要重新执行

```shell
python setup.py build
python setup.py install
```

果然， 只要再执行一遍就不会报 segmentation fault 了，神奇。

另外每次退出虚拟环境之后重新进入也要重新走一套流程。

写着写着又发现了一个很坑的点： matrix.c 里面的函数都是返回 0 表示执行成功， PyObject_TypeCheck 反之。

#### Number Method

这部分就是重载 + - * ** 之类的运算符， 还是比较好写的。

大致流程：首先我写好重载规则， 也就是 python 里面的运算符对应哪些 C 函数：

```c
static PyNumberMethods Matrix61c_as_number = {
    /* TODO: YOUR CODE HERE */
    .nb_add = (binaryfunc)Matrix61c_add,
    .nb_subtract = (binaryfunc)Matrix61c_sub,
    .nb_multiply = (binaryfunc)Matrix61c_multiply,
    .nb_negative = (unaryfunc)Matrix61c_neg,
    .nb_absolute = (unaryfunc)Matrix61c_abs,
    .nb_power = (ternaryfunc)Matrix61c_pow,
};
```

然后拿 Matrix61_add 为例吧， 就直接调用 matrix.c 里的函数来写：

```c

/*
 * Adds two numc.Matrix (Matrix61c) objects together. The first operand is self, and
 * the second operand can be obtained by casting `args`.
 * You will have to check if the arguments' dimensions match and if the second operand is an
 * instance of Matrix61c, and throw a type error if anything is violated.
 */
static PyObject *Matrix61c_add(Matrix61c* self, PyObject* args) 
{
    /* TODO: YOUR CODE HERE */
    Matrix61c * other = (Matrix61c *) args; // cast the args to a Matrix
    if (!PyObject_TypeCheck(other, &Matrix61cType)) // check if the cast was successful
    {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
    matrix * result_matrix = NULL;
    if(allocate_matrix(&result_matrix, self -> mat -> rows, self -> mat -> cols))
     // check if the matrix was allocated successfully
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate matrix");
        return NULL;
    }
    if(add_matrix(result_matrix, self -> mat, other -> mat))
     // check if the matrix was added successfully
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to add matrix");
        return NULL;
    }
    Matrix61c * result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result -> mat = result_matrix;
    result -> shape = PyTuple_Pack(2, PyLong_FromLong(result_matrix -> rows), PyLong_FromLong(result_matrix -> cols));
    return (PyObject *) result;
}
```

(可以看到 runtime error 原来都是包的设计者手写的 hhh)

其他的都类似， 就不赘述啦

#### 矩阵运算加速

写完了接口和衔接部分， 接下来来写一下矩阵运算加速的部分。这里的加速技巧涉及：

+ 循环展开
+ 函数内联
+ 向量化 SIMD
+ OpenMP

性能调优也是我做这个 lab 的最大动力（其次是了解 python 如何用 C 写底层）

首先， 我把原来的版本复制到了 matrix_old.h 里面， 作为 baseline 参考。

先来看看不作任何优化的版本的速度：（矩阵为1000 x 1000）

```shell
Matrix Addition: 0.001604 seconds
Matrix Multiplication: 2.465514 seconds
Matrix Power: 4.744177 seconds
Matrix Negation: 0.001320 seconds
Matrix Absolute: 0.001466 seconds
```


然后来看看开了 -O3 编译优化之后：

```shell
Matrix Addition: 0.000815 seconds
Matrix Multiplication: 0.617334 seconds
Matrix Power: 1.238319 seconds
Matrix Negation: 0.000450 seconds
Matrix Absolute: 0.000429 seconds
```

看来 -O3 是真的强， 很多都优化到 1/3 - 1/4 了。
-O3 会做一些函数内联、 循环展开、 向量化之类的工作。 不过编译器肯定做的优化不是最好的， 我可以自己再试试写这些优化。 下方都是关闭编译器优化后的运行时间。

##### 函数内联

我们要内联什么样的函数？ 首先， 函数调用是有开销的， 要维护内存里的栈。 所以对于小型函数和频繁调用的函数， 我们可以采取函数内联的策略加速代码。
基于这个原则，我内联了小型函数 get / set， 和频繁调用的函数 allocate / deallocate。 还是要来看一下实战效果：

``` shell
testing old matrix
Matrix Addition: 0.001689 seconds
Matrix Multiplication: 2.686311 seconds
Matrix Power: 5.069591 seconds
Matrix Negation: 0.001357 seconds
Matrix Absolute: 0.001581 seconds
testing new matrix
Matrix Addition: 0.002542 seconds
Matrix Multiplication: 2.514559 seconds
Matrix Power: 5.103641 seconds
Matrix Negation: 0.001718 seconds
Matrix Absolute: 0.001812 seconds
```

oh no, 效果好像不咋样. 所以我把 allocate 和 deallocate 的内联取消了。

为什么这里内联之后代码反而会变慢？ 因为 allocate/deallocate 都是很大的函数， 内联它会导致代码体积变大， 影响代码加载时间和存储空间， 同时也会导致更多指令被加载到指令缓存中， 使得指令缓存命中率下降。 

##### 循环展开

最开始把矩阵乘法展开了一层, 看看效果

```shell
testing new matrix
Matrix Addition: 0.002325 seconds
Matrix Multiplication: 3.848300 seconds
Matrix Power: 7.677049 seconds
Matrix Negation: 0.002471 seconds
Matrix Absolute: 0.002737 seconds
```

好像效果不咋样TAT， 可能是循环展开影响了指令的局部性

突然发现最开始还是把矩阵乘法顺序写错了。。应该是 ikj 顺序最快(每次算第 i 行 j 列的目标元素， 要算 k 个点积) 但是换个角度来看， 优化空间变大了hhhh 现在就来优化一下循环顺序（循环互换）

我改了一下， 

```c
    int m = mat1 -> rows;
    int n = mat1 -> cols;
    int k = mat2 -> cols;
    for(int i = 0; i < m; i++)
    {
        for(int l = 0; l < n; l++)
        {
            double temp = mat1 -> data[i * n + l];
            for(int j = 0; j < k; j += 2)
            {
                result -> data[i * k + j] += temp * mat2 -> data[l * k + j];
            }
        }
    }
```

这下变快了一点点：

```shell
testing new matrix
Matrix Addition: 0.002874 seconds
Matrix Multiplication: 3.294082 seconds
Matrix Power: 6.521465 seconds
Matrix Negation: 0.003110 seconds
Matrix Absolute: 0.002464 seconds
```

##### 向量化



##### 并行计算

接下来捣鼓了一下 OpenMP, 我直接在内部套了个 omp for:

```c
    for(int i = 0; i < m; i++)
    {
        for(int l = 0; l < n; l++)
        {
            double temp = mat1 -> data[i * n + l];
            # pragma omp parallel for
            for(int j = 0; j < k; j++)
            {
                result -> data[i * k + j] += temp * mat2 -> data[l * k + j];
            }
        }
    }
```

加速效果大概是。。。。。。。。。。。。

```shell
Matrix Addition: 0.002481 seconds
Matrix Multiplication: 348.358267 seconds
```

只能说是灾难。。

后来想明白了， 这样内部循环加 omp 的话， 每一次走到内部循环都启动一个并行区域， 这样开销实在是太大了。 所以 omp 用在外部循环才好。

然后写了下外部循环的 omp:

```c
    # pragma omp parallel for
    for(int i = 0; i < m; i++)
    {
        for(int l = 0; l < n; l++)
        {
            double temp = mat1 -> data[i * n + l];
            for(int j = 0; j < k; j++)
            {
                result -> data[i * k + j] += temp * mat2 -> data[l * k + j];
            }
        }
    }
```

测了下结果：

``` shell
testing new matrix
Matrix Addition: 0.002258 seconds
Matrix Multiplication: 6.555411 seconds
Matrix Power: 11.381834 seconds
Matrix Negation: 0.021555 seconds
Matrix Absolute: 0.001811 seconds
```

好吧， 还是负优化了TAT

