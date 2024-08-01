include "table.mzn";
include "minicpp.mzn";

var 1..10 : AnohterxNotInTable;
var 1..10 : x0;
var 30..50 : x1;
var 5..50 : x2;
var 1..50 : x3;
var 10..20 : xNotInTable;

array [int,int] of 1..50: t1 = [|1,30,8|2,32,9|4,32,9|2,31,9|4,36,9|4,37,9|];
constraint table([x0,x1,x2], t1) ::gpu;
array [int,int] of 1..50: t2 = [|4,35,9|4,38,9|];
constraint table([x0,x1,x2], t2) ::gpu;
constraint x1>30;
constraint x1<40;
constraint x2>=8;
constraint x3<=35;
solve satisfy;