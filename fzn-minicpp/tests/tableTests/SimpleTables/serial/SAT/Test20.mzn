include "table.mzn";
include "minicpp.mzn";
include "decreasing.mzn";

var 1..10: x0;
var 34..100 : x1;
var 51..100 : x2;
array[1..10] of var 33..55: x; 
array [int,int] of 1..100: t = [|1,39,81|2,38,91|4,37,91|];
array [int,int] of 1..55: t2 = [|33,34,35|40,39,40|53,40,33|];
constraint table([x0,x1,x2], t) ::uniud;
constraint table([x[2],x[3],x[4]], t2) ::uniud;
constraint decreasing(x);
constraint x1>36;
solve satisfy;