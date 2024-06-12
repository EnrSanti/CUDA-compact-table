include "table.mzn";
include "minicpp.mzn";

var 10..20 : AnohterxNotInTable;
var 1..10 : x0;
var 100..300 : x1;
var 100..300 : x2;

array [int,int] of 1..300: t = [|1,139,181|2,138,191|4,137,191|];

constraint table([x0,x1,x2], t) ::gpu;
constraint x1>134;
constraint x2>150;
solve satisfy;