include "table.mzn";
include "minicpp.mzn";

var 1..10 : x0;
var 30..50 : x1;
var 5..50 : x2;
var 5..50 : x3;

array [int,int] of 1..50: t = [|1,30,9|10,33,9|4,35,9|];
array [int,int] of SmartTableOp: signs = [|Int,All,Int|Int,All,Int|Int,Int,Int|];

array [int,int] of 1..50: t2 = [|31,30,8|40,33,9|45,40,8|47,40,9|];
  array [int,int] of SmartTableOp: signs2 = [|Int,Int,Int|Int,Int,Int|Int,All,Int|];

constraint smart_table([x0,x1,x2], t, signs);
constraint smart_table([x1,x2,x3], t2, signs2);

constraint x3>8;
solve satisfy;
