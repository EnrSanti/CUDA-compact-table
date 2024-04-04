#include <stdio.h>
#include "../Shared/shared.h"



void main(int argc, char const* argv[]) {

    //if the user didn't insert the file path or typed more
    if (argc != 2) {
        printf("Insert the file path\n");
        return;
    }
    CT data=readFile(argv[1]);
    //create the strucure
    
}