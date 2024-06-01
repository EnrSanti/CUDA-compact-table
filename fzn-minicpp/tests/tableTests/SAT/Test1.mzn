include "table.mzn";
include "minicpp.mzn";

var 10..20 : AnohterxNotInTable;
var 1..10 : x0;
var 34..100 : x1;
var 51..100 : x2;
var 10..33 : xNotInTable;

array [int,int] of 1..100: t = [|1,39,81|2,38,91|4,37,91|];

constraint table([x0,x1,x2], t) ::uniud;
constraint xNotInTable>15;
constraint x1>35;
solve satisfy;