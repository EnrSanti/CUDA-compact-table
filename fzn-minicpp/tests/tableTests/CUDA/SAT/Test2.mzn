include "table.mzn";
include "minicpp.mzn";

var 1..10 : x0;
var 1..10 : x1;
var 1..10 : x2;

array [int,int] of 1..10: t = [|1,2,8|3,2,9|4,2,10|5,2,10|];

constraint table([x0,x1,x2], t) ::gpu;

solve satisfy;