void main(int argc, char const* argv[]) {

    //if the user didn't insert the file path or typed more
    if (argc != 2) {
        printError("Insert the file path");
        return;
    }
    //create the strucure
    
    //we populate it with the data from the file
    if(readFile_allocateMatrix(argv[1], &data)==-1){
    	return;
    }
    //print the matrix
    //printMatrix(data.matrix);
    printf("\n");
}