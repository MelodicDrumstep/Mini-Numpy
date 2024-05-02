#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>
#include "matrix.h"


// 检查两个矩阵是否相等
bool compare_matrices(matrix *mat1, matrix *mat2, double tolerance) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return false;
    }
    for (int i = 0; i < mat1->rows * mat1->cols; i++) {
        if (fabs(mat1->data[i] - mat2->data[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// 简单的矩阵乘法
void simple_mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    fill_matrix(result, 0.0);
    for (int i = 0; i < mat1->rows; i++) {
        for (int k = 0; k < mat2->cols; k++) {
            for (int j = 0; j < mat1->cols; j++) {
                result->data[i * mat2->cols + k] += mat1->data[i * mat1->cols + j] * mat2->data[j * mat2->cols + k];
            }
        }
    }
}

int main() {
    // 假设已定义 matrix 结构体和相关的 AVX 加速乘法函数 mul_matrix

    // 创建和初始化矩阵
    matrix A = {2, 3, (double[]) {1, 2, 3, 4, 5, 6}};
    matrix B = {3, 2, (double[]) {7, 8, 9, 10, 11, 12}};
    matrix C = {2, 2, malloc(4 * sizeof(double))};
    matrix D = {2, 2, malloc(4 * sizeof(double))};

    // 执行矩阵乘法
    simple_mul_matrix(&D, &A, &B);  // 使用简单方法计算
    mul_matrix(&C, &A, &B);        // 使用 AVX 加速方法计算

    // 比较两种方法的结果
    if (compare_matrices(&C, &D, 0.001)) {
        printf("The matrices are equal.\n");
    } else {
        printf("The matrices are NOT equal.\n");
    }

    // 释放内存
    free(C.data);
    free(D.data);

    return 0;
}
