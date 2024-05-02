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

剩下的循环展开我都实现在向量化之后， 在向量化这一模块里面会提及。
感叹一下向量化 + 循环展开才是真正的大杀器......

##### 并行计算

接下来捣鼓了一下 OpenMP， 这里面有部分是向量化之前写的， 有部分是向量化之后写的

###### 矩阵加法

对于矩阵加法在向量化基础上继续上 OpenMP:

```c
    #pragma omp parallel for
    for(int i = 0; i < mat1 -> rows * mat1 -> cols / 4; i++)
    {
        avx_register1 = _mm256_loadu_pd(mat1 -> data + i * 4);
        // load the data from mat1
        avx_register2 = _mm256_loadu_pd(mat2 -> data + i * 4);
        __m256d avx_result = _mm256_add_pd(avx_register1, avx_register2);
        _mm256_storeu_pd(result -> data + i * 4, avx_result);
        // store the avx_register to the data
    }
    #pragma omp parallel for
    for(int i = mat1 -> rows * mat1 -> cols / 4 * 4; i < mat1 -> rows * mat1 -> cols; i++)
    {
        // For the remaining part
        // use regular way to set the value
        result -> data[i] = mat1 -> data[i] + mat2 -> data[i];
    }
    return 0;
```

看下效果。。。呃

```shell
testing old matrix
Matrix Addition: 0.003440 seconds
...
testing new matrix
Matrix Addition: 0.794540 seconds
```

pardon? 怎么加了 OpenMP 慢了这么多， 合理怀疑是 i 跳步导致缓存一直打不中了...

我把矩阵该到了 10000 x 10000， 这下差距更显著了：

```shell
testing old matrix
Matrix Addition: 0.262105 seconds
testing new matrix
Matrix Addition: 15.001568 seconds
```

我分析了一下， 首先， 第二个循环这部分这么小的数量不应该并行， 把这个去掉， 发现果然性能好了很多， 因为开并行线程是需要开销的

```shell
testing old matrix
Matrix Addition: 0.264391 seconds
testing new matrix
Matrix Addition: 5.237494 seconds
```

其次呢， 我觉得可能是因为向量寄存器是有限的， 一般可能只有 8 个吧， 那这么多线程显然资源就不够用了， 编译器可能要生成好多额外指令来处理超出寄存器容量的部分。 所以这里确实不适合用并行优化， 我把这里的 omp 去掉了。

###### 矩阵乘法

对于矩阵乘法我先没搞 avx 向量化， 先直接在内部套了个 omp for:

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

好吧， 还是负优化了TAT 这部分先搞下 avx 向量化优化， 具体写在 “向量化” 模块里。


##### 向量化

大致就是用 SIMD 指令。 这里我用的是 intel avx extension。

###### 矩阵加法

比如这个 add_matrix, 以前是一个一个算， 现在可以四个四个算
（因为可以直接放到向量寄存器里面）

```c
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if(result -> rows != mat1 -> rows || result -> cols != mat1 -> cols || result -> rows != mat2 -> rows || result -> cols != mat2 -> cols)
    {

        //DEBUG
        #ifdef DEBUG2
        printf("fail");
        #endif 
        //DEBUG

        return -1;
    }
    __m256d avx_register;
    // create a avx_register

    for(int i = 0; i < mat1 -> rows * mat1 -> cols / 4; i++)
    {
        avx_register = _mm256_set_pd(mat1 -> data[i * 4] + mat2 -> data[i * 4],
                                     mat1 -> data[i * 4 + 1] + mat2 -> data[i * 4 + 1],
                                     mat1 -> data[i * 4 + 2] + mat2 -> data[i * 4 + 2],
                                     mat1 -> data[i * 4 + 3] + mat2 -> data[i * 4 + 3]);
        // set the avx_register with 4 val
        _mm256_storeu_pd(result -> data + i * 4, avx_register);
        // store the avx_register to the data
    }
    for(int i = mat1 -> rows * mat1 -> cols / 4 * 4; i < mat1 -> rows * mat1 -> cols; i++)
    {
        // For the remaining part
        // use regular way to set the value
        result -> data[i] = mat1 -> data[i] + mat2 -> data[i];
    }
    return 0;
}
```

很奇怪的是， 这个加速效果没有我想象中的那么好

```c
testing new matrix
Matrix Addition: 0.002880 seconds
```

后来我明白了， 这里每轮循环不要把连续的 4 个元素 load 4 次，那样多浪费啊， 应该直接一起 load 的。 这个可以用 avx 对应的指令。

所以应该写成这样：

``` c
    for(int i = 0; i < mat1 -> rows * mat1 -> cols / 4; i++)
    {
        avx_register1 = _mm256_loadu_pd(mat1 -> data + i * 4);
        // load the data from mat1
        avx_register2 = _mm256_loadu_pd(mat2 -> data + i * 4);
        __m256d avx_result = _mm256_add_pd(avx_register1, avx_register2);
        _mm256_storeu_pd(result -> data + i * 4, avx_result);
        // store the avx_register to the data
    }
    for(int i = mat1 -> rows * mat1 -> cols / 4 * 4; i < mat1 -> rows * mat1 -> cols; i++)
    {
        // For the remaining part
        // use regular way to set the value
        result -> data[i] = mat1 -> data[i] + mat2 -> data[i];
    }
    return 0;
```

可以更加模块化一点写成这样， 方便之后循环展开修改：

```c
__m256d avx_register1;
__m256d avx_register2;
__m256d avx_result;
// create avx_registers

//#pragma omp parallel for

int num_element = mat1 -> rows * mat1 -> cols;
int max_index = num_element / 4 * 4;

for(int i = 0; i < max_index / 4; i++)
{
    avx_register1 = _mm256_loadu_pd(mat1 -> data + i * 4);
    // load the data from mat1
    avx_register2 = _mm256_loadu_pd(mat2 -> data + i * 4);
    avx_result = _mm256_add_pd(avx_register1, avx_register2);
    _mm256_storeu_pd(result -> data + i * 4, avx_result);
    // store the avx_register to the data
}
for(int i = max_index; i < num_element; i++)
{
    // For the remaining part
    // use regular way to set the value
    result -> data[i] = mat1 -> data[i] + mat2 -> data[i];
}
return 0;
```

看下效果：

``` shell
testing old matrix
Matrix Addition: 0.002406 seconds
...
testing new matrix
Matrix Addition: 0.001581 seconds
```

感觉还不错!

###### 矩阵绝对值

像很多操作用 avx 加速都和矩阵加法神似， 这里就不多赘述了。 比较有意思的是这个： 矩阵元素绝对值化：

如果要上 avx， 可以写成这样：

```c
int abs_matrix(matrix *result, matrix *mat) {
    if(result -> rows != mat -> rows || result -> cols != mat -> cols)
    {
        return -1;
    }
    __m256d avx_register;
    __m256d sign_mask;
    __m256d abs_vec;
    //#pragma omp parallel for
    for(int i = 0; i < mat -> rows * mat -> cols / 4; i ++)
    {
        avx_register = _mm256_loadu_pd(mat -> data + i * 4);
        sign_mask = _mm256_set1_pd(-0.0);
        //This represents 100000...0
        abs_vec = _mm256_andnot_pd(sign_mask, avx_register);
        //This will take not of A and compute ((not A) and B)
        // Thus "not A" is 0111....1
        // And the effect is clear the sign bit of avx_register if it's 1
        // meaning if it's negative, make it positive
        _mm256_storeu_pd(result -> data + i * 4, abs_vec);
    }

    for(int i = mat -> rows * mat -> cols / 4 * 4; i < mat -> rows * mat -> cols; i++)
    {
        result -> data[i] = abs(mat -> data[i]);
    }
    return 0;
}
```

这里是怎么用 avx 实现绝对值的？ 这就涉及到 IEEE 对于 double 的存储规定了。double 是第一位先存符号位， 0 表示正数， 1 表示负数， 之后再存指数， 最后存小数。 所以我只需要动第一位， 也就是符号位就行了。

所以我这样做：我先用 `_mm256_set1_pd(-0.0)` 搞出一个 100...0 的东西（负0）， 然后我把它取反， 再和我的数据 and 上。

另外注意， 这里要写 `_mm256_storeu_pd` 而不是 `_mm256_store_pd`。
后者要求数据内存对齐， 性能更高， 但是这里我数据内存不一定对齐， 所以用 `_mm256_store_pd` 会 segmentation fault.

看下效果， 感觉还不错：

```shell
testing old matrix
Matrix Addition: 0.002653 seconds
Matrix Negation: 0.002575 seconds
Matrix Absolute: 0.003015 seconds
testing new matrix
Matrix Addition: 0.001634 seconds
Matrix Negation: 0.002097 seconds
Matrix Absolute: 0.001436 seconds
```

###### 矩阵相反数

然后这个矩阵元素取相反数实现就很简单， 先搞出 0， 然后用 0 - data.

```c
    __m256d avx_register;
    // Create an AVX register filled with zeros
    __m256d avx_zero = _mm256_setzero_pd();
    __m256d avx_result;
    //#pragma omp parallel for
    for(int i = 0; i < mat -> rows * mat -> cols / 4; i ++)
    {

        avx_register = _mm256_loadu_pd(mat -> data + i * 4);
        // Compute the negative value
        avx_result = _mm256_sub_pd(avx_zero, avx_register);
        _mm256_storeu_pd(result -> data + i * 4, avx_result);
    }

    for(int i = mat -> rows * mat -> cols / 4 * 4; i < mat -> rows * mat -> cols; i++)
    {
        result -> data[i] = -mat -> data[i];
    }
    return 0;
```

加速效果蛮不错：

```shell
testing old matrix
Matrix Addition: 0.001721 seconds
Matrix Negation: 0.001434 seconds
Matrix Absolute: 0.001775 seconds
testing new matrix
Matrix Addition: 0.001090 seconds
Matrix Negation: 0.000832 seconds
Matrix Absolute: 0.001282 seconds
```

##### 向量化 + 循环展开

###### 矩阵加法

先来展开一下矩阵加法， 我的测试正确性的 .cpp 文件在 /testing/usertest.c 里。

首先先展开一层：

```c
int num_element = mat1 -> rows * mat1 -> cols;
int max_index = num_element / 8;

for(int i = 0; i < max_index; i += 2)
{
    avx_register1 = _mm256_loadu_pd(mat1 -> data + i * 4);
    // load the data from mat1
    avx_register2 = _mm256_loadu_pd(mat2 -> data + i * 4);
    avx_result = _mm256_add_pd(avx_register1, avx_register2);
    _mm256_storeu_pd(result -> data + i * 4, avx_result);
    // store the avx_register to the data

    avx_register1 = _mm256_loadu_pd(mat1 -> data + (i + 1) * 4);
    // load the data from mat1
    avx_register2 = _mm256_loadu_pd(mat2 -> data + (i + 1) * 4);
    avx_result = _mm256_add_pd(avx_register1, avx_register2);
    _mm256_storeu_pd(result -> data + (i + 1) * 4, avx_result);
}
for(int i = max_index; i < num_element; i++)
{
    // For the remaining part
    // use regular way to set the value
    result -> data[i] = mat1 -> data[i] + mat2 -> data[i];
}
```

这里要注意 i 每次变化多少， 以及最后剩余的部分还有多少。 最好是先定义好这样的 num_element 和 max_index， 免得有时候忘记改 max_index 那个东西了。

看下效果：

```shell
This is -O3 Optimization
Matrix Addition: 0.001031 seconds
testing new matrix
Matrix Addition: 0.003074 seconds
```

已经很接近 -O3 了。 

现在我有一个疑问：要不要把 i++ 打进去？ 比如， 我写成

``` c
for(int i = 0; i < max_index;)
{
    avx_register1 = _mm256_loadu_pd(mat1 -> data + i * 4);
    // load the data from mat1
    avx_register2 = _mm256_loadu_pd(mat2 -> data + i * 4);
    avx_result = _mm256_add_pd(avx_register1, avx_register2);
    _mm256_storeu_pd(result -> data + i * 4, avx_result);
    // store the avx_register to the data

    i++;

    avx_register1 = _mm256_loadu_pd(mat1 -> data + i * 4);
    // load the data from mat1
    avx_register2 = _mm256_loadu_pd(mat2 -> data + i * 4);
    avx_result = _mm256_add_pd(avx_register1, avx_register2);
    _mm256_storeu_pd(result -> data + i * 4, avx_result);

    i++;
}
```

这样的话正确性是没问题， 如果打进去， 那我每次不用重复算 i + 1 的值了， 但是 store 到 i 这个命令也要 store 两次（原先是一次）

话不多说， 先看下效果：

```shell
This is -O3 Optimization
Matrix Addition: 0.001181 seconds
testing new matrix
Matrix Addition: 0.002503 seconds
```

OK， 现在是比没把 i++ 打进去的版本要快。 说明此时为了不重复计算 i++， 宁愿多一次 store 指令。

下面继续展开试试，列出了一个表格

```
展开层数    没有把 i++ 打进循环的测试时间     把 i++ 打进循环的测试时间
1               0.003074                    0.002503
2               0.001862                    0.002339
3               0.002067                    0.001973
```

所以最快的是， 展开 2 层并不把 i++ 打进循环。


###### 矩阵乘法

假设是 C = A x B, 

A is m x n

B is n x k

C is m x k

我先直接把原来的写法向量化， 原来是：

```c
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

显然内层循环可以向量化, 我直接写成

``` c
for(int l = 0; l < n; l++)
{
    double temp = mat1 -> data[i * n + l];
    __m256d avx_resister1 = _mm256_set1_pd(temp);
    __m256d avx_resister2;
    __m256d avx_resister3;
    for(int j = 0; j < k / 4; j++)
    {
        avx_resister2 = _mm256_loadu_pd(mat2 -> data + l * k + j * 4);
        //result -> data[i * k + j] += temp * mat2 -> data[l * k + j];
        avx_resister3 = _mm256_mul_pd(avx_resister1, avx_resister2);
        _mm256_storeu_pd(result -> data + i * k + j * 4, _mm256_add_pd(_mm256_loadu_pd(result -> data + i * k + j * 4), avx_resister3));
    }

    for(int j = k / 4 * 4; j < k; j++)
    {
        result -> data[i * k + j] += temp * mat2 -> data[l * k + j];
    }
}
```

我们来看看 -O3 的情况（每次机子的状况都不太一样， 所以不好直接用之前的结果）

```shell
This is -O3 Optimization
Matrix Addition: 0.001331 seconds
Matrix Multiplication: 0.701121 seconds
Matrix Power: 1.201645 seconds
Matrix Negation: 0.000474 seconds
Matrix Absolute: 0.000408 seconds
```

对于矩阵乘法，还有一种思路是， 我循环顺序采用 mkn， 但是先把 B 转置。

为什么想到转置 B？ 首先， 矩阵乘法再怎么样也是 $O(n^3)$ 的（不考虑那个奇奇怪怪的分治算法， 那个算法本身常数太大了效果也不好， 工业界用的应该都是 $O(n^3)$的）， 而转置是 $O(n^2)$ 的， $n$ 很大的时候转置所需时间可以忽略。其次， 我如果横着扫描 $A$， 那就必定竖着扫描 $B$， 这时我把 B 转置的话， 我就是横着扫描 $B^T$。

这种做法我简单试了一下， 效果远不如之前的 mnk 顺序 + avx 向量化要好。

下面试一下给矩阵乘法循环展开， 

首先是展开一层， 把 j++ 打进循环：

```c
double temp = mat1 -> data[i * n + l];
__m256d avx_resister1 = _mm256_set1_pd(temp);
__m256d avx_resister2;
__m256d avx_resister3;
int max_index = k / 8 * 8;

for(int j = 0; j < max_index / 4;)
{
    avx_resister2 = _mm256_loadu_pd(mat2 -> data + l * k + j * 4);
    //result -> data[i * k + j] += temp * mat2 -> data[l * k + j];
    avx_resister3 = _mm256_mul_pd(avx_resister1, avx_resister2);
    _mm256_storeu_pd(result -> data + i * k + j * 4, _mm256_add_pd(_mm256_loadu_pd(result -> data + i * k + j * 4), avx_resister3));
    
    j++;
    avx_resister2 = _mm256_loadu_pd(mat2 -> data + l * k + j * 4);
    avx_resister3 = _mm256_mul_pd(avx_resister1, avx_resister2);
    _mm256_storeu_pd(result -> data + i * k + j * 4, _mm256_add_pd(_mm256_loadu_pd(result -> data + i * k + j * 4), avx_resister3));

    j++;
}

for(int j = max_index; j < k; j++)
{
    result -> data[i * k + j] += temp * mat2 -> data[l * k + j];
}
```


```shell
This is -O3 Optimization
Matrix Multiplication: 0.683357 seconds
Matrix Power: 1.260901 seconds
testing new matrix
Matrix Multiplication: 1.176709 seconds
Matrix Power: 2.194075 seconds
```

然后试试不打进循环的展开一层：

```c
double temp = mat1 -> data[i * n + l];
__m256d avx_resister1 = _mm256_set1_pd(temp);
__m256d avx_resister2;
__m256d avx_resister3;
int max_index = k / 8 * 8;

for(int j = 0; j < max_index / 4; j += 2)
{
    avx_resister2 = _mm256_loadu_pd(mat2 -> data + l * k + j * 4);
    //result -> data[i * k + j] += temp * mat2 -> data[l * k + j];
    avx_resister3 = _mm256_mul_pd(avx_resister1, avx_resister2);
    _mm256_storeu_pd(result -> data + i * k + j * 4, _mm256_add_pd(_mm256_loadu_pd(result -> data + i * k + j * 4), avx_resister3));

    avx_resister2 = _mm256_loadu_pd(mat2 -> data + l * k + (j + 1) * 4);
    avx_resister3 = _mm256_mul_pd(avx_resister1, avx_resister2);
    _mm256_storeu_pd(result -> data + i * k + (j + 1) * 4, _mm256_add_pd(_mm256_loadu_pd(result -> data + i * k + (j + 1) * 4), avx_resister3));
}

for(int j = max_index; j < k; j++)
{
    result -> data[i * k + j] += temp * mat2 -> data[l * k + j];
}
```

看看效果：

```shell
This is -O3 Optimization
Matrix Addition: 0.001412 seconds
Matrix Multiplication: 0.770085 seconds
Matrix Power: 1.528365 seconds
Matrix Addition: 0.001900 seconds
Matrix Multiplication: 1.256015 seconds
Matrix Power: 2.413503 seconds
```

这个好像性能好一点！

其他的矩阵操作都可以类似循环展开， 这里就不赘述了。

# 总结

所以我最后跑出的结果是这样的:

(如果您想运行我的性能测试脚本， 您可以执行 `./exp_speed.sh`)


```shell
testing the correctness of the new one
Addition test passed.
Subtraction test passed.
Absolute value test passed.
Fill matrix test passed.
Negation test passed.
Multiplication test passed.
Power test passed.

testing old matrix
Matrix Addition: 0.001653 seconds
Matrix Multiplication: 2.568522 seconds
Matrix Power: 5.446699 seconds
Matrix Negation: 0.001792 seconds
Matrix Absolute: 0.001848 seconds

testing new matrix
Matrix Addition: 0.001640 seconds
Matrix Multiplication: 1.072721 seconds
Matrix Power: 2.048053 seconds
Matrix Negation: 0.001170 seconds
Matrix Absolute: 0.001244 seconds

This is -O3 Optimization of the old one
Matrix Addition: 0.001048 seconds
Matrix Multiplication: 0.676902 seconds
Matrix Power: 1.265835 seconds
Matrix Negation: 0.000657 seconds
Matrix Absolute: 0.000552 seconds

This is -O3 Optimization of the new one
Matrix Addition: 0.000786 seconds
Matrix Multiplication: 0.238256 seconds
Matrix Power: 0.426238 seconds
Matrix Negation: 0.000544 seconds
Matrix Absolute: 0.000374 seconds
```

也就是，我优化过的矩阵运算优化程度如下：

```shell
优化前后都不加 -O3 优化的版本：
矩阵操作                            加速倍数                  
Matrix Addition:                   1.01
Matrix Multiplication:             2.40
Matrix Power:                      2.66
Matrix Negation:                   1.53
Matrix Absolute:                   1.49 

优化前后都加 -O3 优化的版本
矩阵操作                            加速倍数             
Matrix Addition:                   1.33
Matrix Multiplication:             2.83
Matrix Power:                      2.97
Matrix Negation:                   1.20
Matrix Absolute:                   1.47
```

如果我们看最后的加速版本与最初的版本对比， 可以得到：

```shell
最终加速程度
矩阵操作                            加速倍数             
Matrix Addition:                   2.10
Matrix Multiplication:             10.78
Matrix Power:                      12.78
Matrix Negation:                   3.30
Matrix Absolute:                   4.94
```

我主要采取的优化手段是 循环互换 + avx 向量化 + 循环展开， 有少部分使用了 OpenMP 和 函数内联。

我真的爱上了性能调优， 看到自己将代码加速了这么多倍， 真的真的太有成就感了。