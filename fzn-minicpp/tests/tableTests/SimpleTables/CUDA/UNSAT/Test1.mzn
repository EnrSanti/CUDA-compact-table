include "table.mzn";
include "minicpp.mzn";

var 1..10 : x0;
var 1..10 : x1;
var 1..10 : x2;

array [int,int] of 1..10: t = [|1,2,8|];

constraint table([x0,x1,x2], t) ::gpu;
constraint x2>8;
solve satisfy;
