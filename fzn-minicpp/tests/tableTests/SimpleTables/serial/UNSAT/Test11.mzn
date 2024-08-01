include "table.mzn";
include "minicpp.mzn";
include "all_equal.mzn";

var 10..20 : AnohterxNotInTable;  
var 1..10: x0;
var 34..500 : x1;
var 51..100 : x2;
var 10..33 : xNotInTable;
array[1..20] of var 1..10: x; 
array [int,int] of 1..500: t = [|1,39,81|2,38,91|4,37,91|];
array [int,int] of 1..10: t2 = [|1,2,3|2,2,4|2,3,2|4,4,2|];

constraint table([x0,x1,x2], t) ::uniud;
constraint table([x[1],x[2],x[3]], t2) ::uniud;
constraint xNotInTable>15;
constraint x1>36;
constraint x2>65;
constraint all_equal(x);
solve satisfy;