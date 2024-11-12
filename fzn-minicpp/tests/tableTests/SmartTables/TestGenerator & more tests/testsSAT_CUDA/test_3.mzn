%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 29..108 : x0;
var 9..53 : x1;
var 23..68 : x2;
var 22..77 : x3;
var 34..102 : x4;
array [int,int] of 9..108 : t0=[|63,25,94|
51,48,65|
45,59,91|
26,50,79|
47,43,74|
24,28,55|]; 
array [int,int] of SmartTableOp: signs0=[|LtInt,LtInt,Int|Int,Int,Int|GtInt,Int,Int|Int,Int,GtInt|All,Int,Int|Int,Int,Int|]; 
constraint smart_table([x2,x3,x4],t0,signs0)::gpu;
constraint x2<30;
solve satisfy;