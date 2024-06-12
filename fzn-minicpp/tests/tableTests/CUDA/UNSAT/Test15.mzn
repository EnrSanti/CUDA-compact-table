include "table.mzn";
include "minicpp.mzn";
include "all_different.mzn";

var 1..10: x0;
var 34..100 : x1;
var 1..100 : x2;
array[1..10] of var 31..100: x; 


array [int,int] of 1..100: t = [|1,39,81|2,38,91|4,37,91|];
array [int,int] of 1..100: t2 = [|33,35,36|];
array [int,int] of 1..100: t3 = [|31,33,35|37,33,36|];
constraint table([x0,x1,x2], t) ::gpu;
constraint table([x[2],x[3],x[4]], t2) ::gpu;
constraint table([x[1],x[2],x[3]], t3) ::gpu;
constraint all_different(x);
constraint x1>36;
constraint x[1]>36;
solve satisfy;