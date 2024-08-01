include "table.mzn";
include "minicpp.mzn";

var 1..10 : x0;
var 1..100 : x1;
var 1..50 : x2;

array [int,int] of 1..100: t = [|1,30,9|5,38,9|1,36,9|];

constraint table([x0,x1,x2], t) ::gpu;

constraint x0=5;
constraint x1=38;
solve satisfy;