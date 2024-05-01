#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

//#define DEBUG
#define DEBUG2

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) 
{
    if(rows < 1 || cols < 1)
    {

        // DEBUG
        #ifdef DEBUG2
        printf("rows or cols is less than 1\n");
        #endif
        // DEBUG

        return -1;
    }
    *mat = (matrix *)malloc(sizeof(matrix));
    if(*mat == NULL)
    {
  
        // DEBUG
        #ifdef DEBUG2
        printf("mat is NULL\n");
        #endif
        // DEBUG

        return -1;
    }
    (*mat) -> rows = rows;
    (*mat) -> cols = cols;
    (*mat) -> ref_cnt = 1;
    (*mat) -> parent = NULL;
    (*mat) -> data = (double *)malloc(sizeof(double) * rows * cols);
    if ((*mat) -> data == NULL) 
    {
        free(*mat);  // Free the allocated matrix struct if data allocation fails
        *mat = NULL;

        // DEBUG
        #ifdef DEBUG2
        printf("data is NULL\n");
        #endif
        // DEBUG

        return -1;
    }
    for(int i = 0; i < rows * cols; i++)
    {
        (*mat) -> data[i] = 0;
    }
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    if(rows < 1 || cols < 1)
    {
        return -1;
    }
    *mat = (matrix *)malloc(sizeof(matrix));
    if(*mat == NULL)
    {
        return -1;
    }
    (*mat) -> rows = rows;
    (*mat) -> cols = cols;
    (*mat) -> data = from -> data + offset;
    (*mat) -> ref_cnt = 1;
    from -> ref_cnt++;
    (*mat) -> parent = from;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    if(mat == NULL) 
    {
        return;
    }
    mat -> ref_cnt--;
    //Whatever it is, I have to drop the refcnt by 1
    
    if(mat -> ref_cnt == 0)
    { //Now refcnt drop to 0, I have to free this mat
        if(mat -> parent != NULL)
        {
            mat -> parent -> ref_cnt--;
            if(mat -> parent -> ref_cnt == 0)
            {
                deallocate_matrix(mat -> parent);
                //recursively apply it mat -> parent
            }
        }
        else
        {
            //If it's not a slice, then I have to free the data
            free(mat -> data);
        }
        //No matter what it is, I have to free the mat * pointer
        free(mat);
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return mat -> data[row * mat -> cols + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat -> data[row * mat -> cols + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    for(int i = 0; i < mat -> rows * mat -> cols; i++)
    {
        mat -> data[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
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
    for(int i = 0; i < mat1 -> rows * mat1 -> cols; i++)
    {
        result -> data[i] = mat1 -> data[i] + mat2 -> data[i];
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
   if(result -> rows != mat1 -> rows || result -> cols != mat1 -> cols || result -> rows != mat2 -> rows || result -> cols != mat2 -> cols)
    {
        return -1;
    }
    for(int i = 0; i < mat1 -> rows * mat1 -> cols; i++)
    {
        result -> data[i] = mat1 -> data[i] - mat2 -> data[i];
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) 
{
    if(mat1 -> cols != mat2 -> rows || result -> rows != mat1 -> rows || result -> cols != mat2 -> cols)
    {
        return -1;
    }
    int m = mat1 -> rows;
    int n = mat1 -> cols;
    int k = mat2 -> cols;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < k; j++)
        {
            double sum = 0;
            for(int l = 0; l < n; l++)
            {
                sum += mat1 -> data[i * n + l] * mat2 -> data[l * k + j];
            }
            result -> data[i * k + j] = sum;
        }
    }
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */

// Use the idea of fast-power algorithm
/*
int fast_pow(int a, int b) {
    int result = 1;  // 最终结果
    int base = a;    // 基数 a

    while (b > 0) {
        if (b % 2 == 1) {
            // 如果 b 是奇数，乘上当前的基数
            result *= base;
        }
        // b 除以 2
        b /= 2;
        // 基数平方
        base *= base;
    }

    return result;
}
*/
int pow_matrix(matrix *result, matrix *mat, int pow) 
{
    #ifdef DEBUG
    printf("Start! pow is : %d\n", pow);
    #endif 

    if(pow < 0)
    {
        return -1;
    }
    if(mat -> rows != mat -> cols)
    {
        //The prerequisite for a power of matrix 
        //it that it's rectangular
        return -1;
    }
    if(result -> rows != mat -> rows || result -> cols != mat -> cols)
    {
        return -1;
    }
    for(int i = 0; i < mat -> rows * mat -> cols; i++)
    {
        if(i % (mat -> cols + 1) == 0)
        {
            result -> data[i] = 1;
        }
        else
        {
            result -> data[i] = 0;
        }
    }
    if(pow == 0)
    {
        return 0;
    }
    matrix * temp;
    if(allocate_matrix(&temp, mat -> rows, mat -> cols) == -1)
    {
        return -1;
    }
    for(int i = 0; i < mat -> rows * mat -> cols; i++)
    {
        temp -> data[i] = mat -> data[i];
    }
    while(pow > 0)
    {
        #ifdef DEBUG
        printf("In loop! pow is : %d\n", pow);
        #endif

        if(pow & 1)
        {
            matrix * temp3;
            if(allocate_matrix(&temp3, result -> rows, result -> cols) == -1)
            {
                return -1;
            }
            mul_matrix(temp3, result, temp);
            for(int i = 0; i < mat -> rows * mat -> cols; i++)
            {
                result -> data[i] = temp3 -> data[i];
            }
            deallocate_matrix(temp3);

            #ifdef DEBUG 
            printf("the result matrix here is : \n" );
            for(int i = 0; i < result -> rows; i++)
            {
                for(int j = 0; j < result -> cols; j++)
                {
                    printf("%lf ", result -> data[i * result -> cols + j]);
                }
                printf("\n");
            }
            #endif 
        }
        pow >>= 1;

        #ifdef DEBUG
        printf("After pow >> = 1! pow is : %d\n", pow);
        #endif 

        if(pow > 0)
        {
            matrix * temp2;
            if(allocate_matrix(&temp2, mat -> rows, mat -> cols) == -1)
            {
                return -1;
            }
            mul_matrix(temp2, temp, temp);
            for(int i = 0; i < mat -> rows * mat -> cols; i++)
            {
                temp -> data[i] = temp2 -> data[i];
            }
            deallocate_matrix(temp2);
        }
    }
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    if(result -> rows != mat -> rows || result -> cols != mat -> cols)
    {
        return -1;
    }
    for(int i = 0; i < mat -> rows * mat -> cols; i++)
    {
        result -> data[i] = -mat -> data[i];
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    if(result -> rows != mat -> rows || result -> cols != mat -> cols)
    {
        return -1;
    }
    for(int i = 0; i < mat -> rows * mat -> cols; i++)
    {
        result -> data[i] = abs(mat -> data[i]);
    }
    return 0;
}

