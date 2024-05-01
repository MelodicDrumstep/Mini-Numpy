#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_old.h" 

#define MATRIX_SIZE 1000 // 假设矩阵大小为1000x1000
#define NUM_EXPERIMENTS 10 // 每个操作的实验次数

// 定义测试函数
void test_performance() {
    matrix *mat1, *mat2, *result;
    clock_t start, end;
    double cpu_time_used;

    // 分配内存并生成测试数据
    allocate_matrix(&mat1, MATRIX_SIZE, MATRIX_SIZE);
    allocate_matrix(&mat2, MATRIX_SIZE, MATRIX_SIZE);
    allocate_matrix(&result, MATRIX_SIZE, MATRIX_SIZE);
    rand_matrix(mat1, 123, 0.0, 1.0);
    rand_matrix(mat2, 456, 0.0, 1.0);

    // 测试矩阵加法性能
    cpu_time_used = 0;
    for (int i = 0; i < NUM_EXPERIMENTS; ++i) {
        start = clock();
        add_matrix(result, mat1, mat2);
        end = clock();
        cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
    }
    cpu_time_used /= NUM_EXPERIMENTS;
    printf("Matrix Addition: %f seconds\n", cpu_time_used);

    // 测试矩阵乘法性能
    cpu_time_used = 0;
    for (int i = 0; i < NUM_EXPERIMENTS; ++i) {
        start = clock();
        mul_matrix(result, mat1, mat2);
        end = clock();
        cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
    }
    cpu_time_used /= NUM_EXPERIMENTS;
    printf("Matrix Multiplication: %f seconds\n", cpu_time_used);

    // 测试矩阵幂运算性能
    cpu_time_used = 0;
    for (int i = 0; i < NUM_EXPERIMENTS; ++i) {
        start = clock();
        pow_matrix(result, mat1, 2);
        end = clock();
        cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
    }
    cpu_time_used /= NUM_EXPERIMENTS;
    printf("Matrix Power: %f seconds\n", cpu_time_used);

    // 测试矩阵取负性能
    cpu_time_used = 0;
    for (int i = 0; i < NUM_EXPERIMENTS; ++i) {
        start = clock();
        neg_matrix(result, mat1);
        end = clock();
        cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
    }
    cpu_time_used /= NUM_EXPERIMENTS;
    printf("Matrix Negation: %f seconds\n", cpu_time_used);

    // 测试矩阵取绝对值性能
    cpu_time_used = 0;
    for (int i = 0; i < NUM_EXPERIMENTS; ++i) {
        start = clock();
        abs_matrix(result, mat1);
        end = clock();
        cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
    }
    cpu_time_used /= NUM_EXPERIMENTS;
    printf("Matrix Absolute: %f seconds\n", cpu_time_used);

    // 释放内存
    deallocate_matrix(mat1);
    deallocate_matrix(mat2);
    deallocate_matrix(result);
}

int main() {
    test_performance();
    return 0;
}
