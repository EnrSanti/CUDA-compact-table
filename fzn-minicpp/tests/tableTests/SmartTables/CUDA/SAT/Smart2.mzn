include "table.mzn";
include "minicpp.mzn";

array [int,int] of int: t  = [|4,2,3|3,4,5|5,6,6|];
array [int,int] of SmartTableOp: signs = [|All,All,Int|Int,All,Int|Int,Int,All|];

var 1..10 : x0;
var 1..10 : x1;
var 1..10 : x2;
constraint smart_table([x0,x1,x2], t, signs):: gpu;

solve satisfy;