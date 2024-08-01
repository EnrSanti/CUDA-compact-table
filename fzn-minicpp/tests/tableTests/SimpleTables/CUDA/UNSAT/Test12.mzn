  
include "table.mzn";
include "minicpp.mzn";
include "decreasing.mzn";

array[1..2] of var 33..40: x; 
array [int,int] of 1..55: t2 = [|33,34|39,40|38,40|];
constraint table([x[1],x[2]], t2) ::gpu;
constraint decreasing(x);
solve satisfy;