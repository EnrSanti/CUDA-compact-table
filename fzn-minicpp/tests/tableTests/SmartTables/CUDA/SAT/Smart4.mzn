include "table.mzn";
include "minicpp.mzn";

array [int,int] of int: t  = [|5,5,3|6,5,1|5,6,6|];
array [int,int] of SmartTableOp: signs = [|All,All,Int|Int,All,Int|Int,Int,Int|];

var 5..10 : x0;
var 5..10 : x1;
var 1..10 : x2;
var 10..20 : xNotInTable;
constraint smart_table([x0,x1,x2], t, signs) :: gpu;

constraint xNotInTable>15;
constraint x1=6; 

solve satisfy;