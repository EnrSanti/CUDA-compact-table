include "table.mzn";
include "minicpp.mzn";

var 1..10 : AnohterxNotInTable;
var 1..10 : x0;
var 30..70 : x1;
var 5..120 : x2;
var 15..100 : x3;
var 10..20 : xNotInTable;

array [int,int] of 1..120: t1 = [|1,30,8|2,32,9|4,32,9|2,31,9|4,36,9|4,37,9|4,37,30|];
constraint table([x0,x1,x2], t1) ::gpu;
array [int,int] of 1..120: t2 = [|30,8,25|36,9,25|37,30,29|37,30,30|];
constraint table([x1,x2,x3], t2) ::gpu;
constraint x1>35;
constraint x2>29;
constraint not(x2=x3);
solve satisfy;