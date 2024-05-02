#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "matrix_old.h"
#include "math.h"

// 生成随机矩阵
void generate_random_matrix(matrix *mat, int rows, int cols) {
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double *)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; i++) {
        mat->data[i] = (double)rand() / RAND_MAX * 100;  // 生成0到100之间的随机数
    }
}

void generate_zero_matrix(matrix *mat, int rows, int cols) {
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double *)calloc(rows * cols, sizeof(double));  // 使用calloc保证初始化为0
}


// 比较两个矩阵是否相等，并打印不一致的元素
int compare_matrices(matrix *mat1, matrix *mat2) {
    int equal = 1;
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        printf("Matrix dimensions mismatch.\n");
        return 0;
    }
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            double val1 = mat1->data[i * mat1->cols + j];
            double val2 = mat2->data[i * mat2->cols + j];
            if (fabs(val1 - val2) > 0.00001) {  // 使用适当的容差
                printf("Mismatch at row %d, col %d: %f != %f\n", i, j, val1, val2);
                equal = 0;
            }
        }
    }
    return equal;
}

// 测试矩阵加法
void test_addition() {
    matrix mat1, mat2, result1, result2;
    generate_random_matrix(&mat1, 100, 100);
    generate_random_matrix(&mat2, 100, 100);
    generate_random_matrix(&result1, 100, 100); // 初始分配结果矩阵
    generate_random_matrix(&result2, 100, 100); // 初始分配结果矩阵

    add_matrix(&result1, &mat1, &mat2);
    add_matrix_old(&result2, &mat1, &mat2);

    if (compare_matrices(&result1, &result2)) {
        printf("Addition test passed.\n");
    } else {
        printf("Addition test failed.\n");
    }

    free(mat1.data);
    free(mat2.data);
    free(result1.data);
    free(result2.data);
}

// 测试填充矩阵函数
void test_fill_matrix() {
    matrix mat1, result1, result2;
    generate_random_matrix(&mat1, 100, 100);
    generate_random_matrix(&result1, 100, 100); // 创建结果矩阵
    generate_random_matrix(&result2, 100, 100); // 创建结果矩阵

    fill_matrix(&result1, 5.5);
    fill_matrix_old(&result2, 5.5);

    if (compare_matrices(&result1, &result2)) {
        printf("Fill matrix test passed.\n");
    } else {
        printf("Fill matrix test failed.\n");
    }

    free(mat1.data);
    free(result1.data);
    free(result2.data);
}

// 测试矩阵减法
void test_subtraction() {
    matrix mat1, mat2, result1, result2;
    generate_random_matrix(&mat1, 100, 100);
    generate_random_matrix(&mat2, 100, 100);
    generate_random_matrix(&result1, 100, 100); // 创建结果矩阵
    generate_random_matrix(&result2, 100, 100); // 创建结果矩阵

    sub_matrix(&result1, &mat1, &mat2);
    sub_matrix_old(&result2, &mat1, &mat2);

    if (compare_matrices(&result1, &result2)) {
        printf("Subtraction test passed.\n");
    } else {
        printf("Subtraction test failed.\n");
    }

    free(mat1.data);
    free(mat2.data);
    free(result1.data);
    free(result2.data);
}

// 测试矩阵元素取绝对值
void test_abs() {
    matrix mat1, result1, result2;
    generate_random_matrix(&mat1, 100, 100);
    generate_zero_matrix(&result1, 100, 100); // 创建结果矩阵
    generate_zero_matrix(&result2, 100, 100); // 创建结果矩阵

    abs_matrix(&result1, &mat1);
    abs_matrix_old(&result2, &mat1);

    if (compare_matrices(&result1, &result2)) {
        printf("Absolute value test passed.\n");
    } else {
        printf("Absolute value test failed.\n");
    }

    // for(int i = 0; i < 100; i++) {
    //     for(int j = 0; j < 100; j++) {
    //         printf("i = %d, j = %d, result1 = %f, result2 = %f\n", i, j, result1.data[i * 100 + j], result2.data[i * 100 + j]);
    //     }
    // }

    free(mat1.data);
    free(result1.data);
    free(result2.data);
}

// 测试矩阵取负
void test_negation() {
    matrix mat, result1, result2;
    generate_random_matrix(&mat, 100, 100);
    generate_random_matrix(&result1, 100, 100); // 创建结果矩阵
    generate_random_matrix(&result2, 100, 100); // 创建结果矩阵

    neg_matrix(&result1, &mat);
    neg_matrix_old(&result2, &mat);

    if (compare_matrices(&result1, &result2)) {
        printf("Negation test passed.\n");
    } else {
        printf("Negation test failed.\n");
    }

    free(mat.data);
    free(result1.data);
    free(result2.data);
}

// 测试矩阵乘法
void test_multiplication() {
    matrix mat1, mat2, result1, result2;
    generate_random_matrix(&mat1, 100, 100);
    generate_random_matrix(&mat2, 100, 100);
    generate_random_matrix(&result1, 100, 100); // 创建结果矩阵
    generate_random_matrix(&result2, 100, 100); // 创建结果矩阵

    mul_matrix(&result1, &mat1, &mat2);
    mul_matrix_old(&result2, &mat1, &mat2);

    if (compare_matrices(&result1, &result2)) {
        printf("Multiplication test passed.\n");
    } else {
        printf("Multiplication test failed.\n");
    }

    free(mat1.data);
    free(mat2.data);
    free(result1.data);
    free(result2.data);
}

// 测试矩阵求幂
void test_power() {
    matrix mat, result1, result2;
    int power = 3;  // 测试幂为3的情况
    generate_random_matrix(&mat, 100, 100);
    generate_random_matrix(&result1, 100, 100); // 创建结果矩阵
    generate_random_matrix(&result2, 100, 100); // 创建结果矩阵

    pow_matrix(&result1, &mat, power);
    pow_matrix_old(&result2, &mat, power);

    if (compare_matrices(&result1, &result2)) {
        printf("Power test passed.\n");
    } else {
        printf("Power test failed.\n");
    }

    free(mat.data);
    free(result1.data);
    free(result2.data);
}

int main() {
    srand(time(NULL));  // 设置随机种子

    test_addition();      // 测试加法
    test_subtraction();   // 测试减法
    test_abs();           // 测试取绝对值
    test_fill_matrix();   // 测试填充矩阵
    test_negation();      // 测试取负
    test_multiplication(); // 测试乘法
    test_power();         // 测试求幂
    return 0;
}
