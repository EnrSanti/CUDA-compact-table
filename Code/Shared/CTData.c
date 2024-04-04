#include <stdio.h>

typedef struct CT{
    char** scope; //variable names
    bitSet currTable;
    char** s_val;
    char** s_sup;
    int* lastSizes;
    bitSet* supports;
    int* residues;
} CT;
