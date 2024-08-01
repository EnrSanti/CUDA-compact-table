include "table.mzn";
include "minicpp.mzn";

var 1..10 : x0;
var 1..10 : x1;
var 5..10 : x2;
var 10..20 : xNotInTable;

array [int,int] of 1..10: t = [|1,8,8|1,2,9|];

constraint table([x0,x1,x2], t) ::uniud;
constraint xNotInTable>15;
constraint x1>5;
constraint x2>5;
constraint x0>5;
solve satisfy;