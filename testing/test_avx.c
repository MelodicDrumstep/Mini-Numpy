#include <stdio.h>
#include <immintrin.h>

union DoubleVec {
    __m256d vec;
    double arr[4];
};

int main() {
    union DoubleVec dv;
    dv.vec = _mm256_set1_pd(-0.0);

    for (int i = 0; i < 4; i++) {
        printf("Element %d: %lf\n", i, dv.arr[i]);
        unsigned char *bytePtr = (unsigned char *)&dv.arr[i];
        printf("Binary: ");
        for (int j = sizeof(double) - 1; j >= 0; j--) {
            for (int k = 7; k >= 0; k--) {
                printf("%d", (bytePtr[j] >> k) & 1);
            }
            printf(" ");
        }
        printf("\n");
    }

    return 0;
}
