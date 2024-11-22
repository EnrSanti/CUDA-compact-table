include "table.mzn";
include "minicpp.mzn";
include "all_equal.mzn";

var 10..33 : xNotInTable;
array[1..20] of var 1..10: x; 
array [int,int] of 1..10: t2 = [|1,5,3|2,2,4|2,3,2|4,4,2|];

constraint table([x[1],x[2],x[3]], t2) ::gpu;
constraint all_equal(x);
solve satisfy;