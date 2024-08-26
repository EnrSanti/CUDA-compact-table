include "table.mzn";
include "minicpp.mzn";

var 10..20 : AnohterxNotInTable;
var 1..10 : x0;
var 30..50 : x1;
var 5..50 : x2;
var 10..20 : xNotInTable;

array [int,int] of 1..50: t = [|1,30,8|2,38,9|4,35,9|];
array [int,int] of SmartTableOp: signs = [|All,All,Int|Int,All,Int|Int,Int,Int|];

constraint smart_table([x0,x1,x2], t, signs) :: gpu;
constraint xNotInTable>15;
constraint x1>35;


solve satisfy;