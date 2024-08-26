include "table.mzn";
include "minicpp.mzn";


var 1..10 : x0;
var 30..50 : x1;
var 5..50 : x2;
array [int,int] of 1..50: t = [|1,30,8|2,32,9|4,32,9|2,50,9|4,36,9|4,45,9|];
array [int,int] of SmartTableOp: signs = [|All,All,Int|All,All,Int|Int,Int,Int|Int,Int,Int|Int,Int,Int|Int,Int,Int|];

constraint smart_table([x0,x1,x2], t, signs) :: gpu;

constraint x1>35;
constraint x1<40;
constraint x2=22;
solve satisfy;