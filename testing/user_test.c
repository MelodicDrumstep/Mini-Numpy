#include "matrix.h"
#include <stdio.h>

// Function to print a matrix
void print_matrix(matrix *mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            printf("%lf ", get(mat, i, j));
        }
        printf("\n");
    }
}

// Function to compare two matrices
int compare_matrices(matrix *mat1, matrix *mat2) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return 0; // Matrices have different dimensions
    }
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            if (get(mat1, i, j) != get(mat2, i, j)) {
                return 0; // Matrices are not equal
            }
        }
    }
    return 1; // Matrices are equal
}

int main() {
    matrix *mat1, *mat2, *result;
    int rows = 3, cols = 3;

    // Test allocate_matrix
    printf("Testing allocate_matrix...\n");
    if (allocate_matrix(&mat1, rows, cols) == 0) {
        printf("allocate_matrix: Success\n");
    } else {
        printf("allocate_matrix: Failed\n");
        return 1;
    }

    // Test rand_matrix
    printf("\nTesting rand_matrix...\n");
    rand_matrix(mat1, 42, 0.0, 1.0);
    print_matrix(mat1);

    // Test add_matrix
    printf("\nTesting add_matrix...\n");
    allocate_matrix(&mat2, rows, cols);
    rand_matrix(mat2, 24, 0.0, 1.0);
    allocate_matrix(&result, rows, cols);
    if (add_matrix(result, mat1, mat2) == 0) {
        printf("add_matrix: Success\n");
        print_matrix(result);
    } else {
        printf("add_matrix: Failed\n");
    }

    // Test sub_matrix
    printf("\nTesting sub_matrix...\n");
    if (sub_matrix(result, mat1, mat2) == 0) {
        printf("sub_matrix: Success\n");
        print_matrix(result);
    } else {
        printf("sub_matrix: Failed\n");
    }

    // Test mul_matrix
    printf("\nTesting mul_matrix...\n");
    matrix *mat3, *mat4;
    allocate_matrix(&mat3, 3, 2);
    allocate_matrix(&mat4, 2, 3);
    rand_matrix(mat3, 15, 0.0, 1.0);
    rand_matrix(mat4, 36, 0.0, 1.0);
    allocate_matrix(&result, 3, 3);
    if (mul_matrix(result, mat3, mat4) == 0) {
        printf("mul_matrix: Success\n");
        print_matrix(result);
    } else {
        printf("mul_matrix: Failed\n");
    }

    // Test pow_matrix
    printf("\nTesting pow_matrix...\n");
    matrix *square_mat;
    allocate_matrix(&square_mat, 3, 3);
    rand_matrix(square_mat, 10, 0.0, 1.0);
    allocate_matrix(&result, 3, 3);
    if (pow_matrix(result, square_mat, 2) == 0) {
        printf("pow_matrix: Success\n");
        print_matrix(result);
    } else {
        printf("pow_matrix: Failed\n");
    }

    // Test neg_matrix
    printf("\nTesting neg_matrix...\n");
    if (neg_matrix(result, mat1) == 0) {
        printf("neg_matrix: Success\n");
        print_matrix(result);
    } else {
        printf("neg_matrix: Failed\n");
    }

    // Test abs_matrix
    printf("\nTesting abs_matrix...\n");
    if (abs_matrix(result, mat1) == 0) {
        printf("abs_matrix: Success\n");
        print_matrix(result);
    } else {
        printf("abs_matrix: Failed\n");
    }

    // Test deallocate_matrix
    printf("\nTesting deallocate_matrix...\n");
    deallocate_matrix(mat1);
    deallocate_matrix(mat2);
    deallocate_matrix(mat3);
    deallocate_matrix(mat4);
    deallocate_matrix(square_mat);
    deallocate_matrix(result);

    return 0;
}
