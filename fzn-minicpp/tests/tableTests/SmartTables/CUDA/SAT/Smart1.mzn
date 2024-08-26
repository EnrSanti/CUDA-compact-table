include "table.mzn";
include "minicpp.mzn";

array [int,int] of int: t  = [|4,2,3|3,4,5|5,6,6|];
array [int,int] of SmartTableOp: signs = [|All,LtInt,Int|Int,GtInt,Int|Int,Int,All|];

var 1..6 : x0;
var 1..6 : x1;
var 1..6 : x2;
constraint smart_table([x0,x1,x2], t, signs):: gpu;

solve satisfy;
