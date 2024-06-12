include "table.mzn";
include "minicpp.mzn";

var 1..10 : AnohterxNotInTable;
var 1..10 : x0;
var 30..50 : x1;
var 5..50 : x2;
var 10..20 : xNotInTable;

array [int,int] of 1..50: t1 = [|1,30,8|2,32,9|4,32,9|2,31,9|4,36,9|4,37,9|];
constraint table([x0,x1,x2], t1) ::uniud;
array [int,int] of 1..50: t2 = [|30,8|37,9|];
constraint table([x1,x2], t2) ::uniud;
constraint x1>35;
constraint AnohterxNotInTable=10;

solve satisfy;