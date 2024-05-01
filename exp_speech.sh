echo "testing the correctness of the new one"
gcc -o test user_test.c matrix.c
./test
echo "test pass!"
# echo "testing old matrix"
# gcc -o sold speed_testing_old.c matrix.c
# ./sold
echo "testing new matrix"
gcc -fopenmp -o snew speed_testing.c matrix.c
./snew