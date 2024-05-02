#ifndef MATRIX_OLD_H
#define MATRIX_OLD_H

#ifndef MATRIX
#define MATRIX

typedef struct matrix {
    int rows; // number of rows
    int cols; // number of columns
    double* data; // pointer to rows * columns doubles
    int ref_cnt; // How many slices/matrices are referring to this matrix's data
    struct matrix *parent; // NULL if matrix is not a slice, else the parent matrix of the slice
} matrix;

#endif

double rand_double_old(double low, double high);
void rand_matrix_old(matrix *result, unsigned int seed, double low, double high);
int allocate_matrix_old(matrix **mat, int rows, int cols);
int allocate_matrix_ref_old(matrix **mat, matrix *from, int offset, int rows, int cols);
void deallocate_matrix_old(matrix *mat);
double get_old(matrix *mat, int row, int col);
void set_old(matrix *mat, int row, int col, double val);
void fill_matrix_old(matrix *mat, double val);
int add_matrix_old(matrix *result, matrix *mat1, matrix *mat2);
int sub_matrix_old(matrix *result, matrix *mat1, matrix *mat2);
int mul_matrix_old(matrix *result, matrix *mat1, matrix *mat2);
int pow_matrix_old(matrix *result, matrix *mat, int pow);
int neg_matrix_old(matrix *result, matrix *mat);
int abs_matrix_old(matrix *result, matrix *mat);

#endif