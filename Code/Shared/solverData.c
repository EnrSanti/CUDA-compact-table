//"solver specific data"
typedef struct solverData{
	int** deltaXs;
	int* deltaXSizes;
	char** domains; //yes, we store explicity in chars the domains (we could use bitsets but this is just a mock-up solver, so why bother) 
	int* domainSizes;
}solverData;
