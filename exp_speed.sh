echo "testing the correctness of the new one"
gcc -mavx -fopenmp -I. -o test testing/user_test.c matrix.c
./test
echo "double test the matrix multiplication"
gcc -mavx -fopenmp -I. -o t testing/test_matmul.c matrix.c
./t
echo "test pass!"
echo "testing old matrix"
gcc -I. -o sold testing/speed_testing_old.c matrix_old.c
./sold
echo "This is -O3 Optimization"
gcc -O3 -I. -o so3 testing/speed_testing_old.c matrix_old.c
./so3
echo "testing new matrix"
gcc -mavx -fopenmp -I. -o snew testing/speed_testing.c matrix.c
./snew
