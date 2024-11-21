#include <stdio.h>

void printBits(unsigned int n) {
    for (int i = 31; i >= 0; i--) {
        printf("%d", (n >> i) & 1);
}
    printf("\n");
}

int main() {
    int a = -16;    // 32-bit signed integer
    unsigned int ua = (unsigned int)a; // Cast ‘a’ to an unsigned integer

    printf("a : ");
    printBits(a);  // Print the 32-bit binary number of ‘a’
    printf("Arithmetic Shift (a >> 2) : ");
    printBits(a >> 2);  // Print the 32-bit binary number of ‘a’ after arithmetic right shift
    printf("Logical Shift (ua >> 2) : ");
    printBits(ua >> 2); // Print the 32-bit binary number of ‘ua’ after logical right shift
return 0; }