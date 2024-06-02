include "table.mzn";
include "minicpp.mzn";


var 1..10: x0;
var 34..100 : x1;
var 51..100 : x2;
var 1..100 : xNotInTable;
var 1..100 : leh;

array [int,int] of 1..100: t = [|1,39,81|2,38,91|4,37,91|];
constraint table([x0,x1,x2], t) ::uniud;

constraint xNotInTable>15;
constraint x1>36;
solve satisfy;