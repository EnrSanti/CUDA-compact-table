include "table.mzn";
include "minicpp.mzn";

var 1..10 : x0;
var 5..10 : x1;
var 1..10 : x2;

array [int,int] of 1..10: t = [|1,5,8|3,6,9|4,7,10|5,5,10|];

constraint table([x0,x1,x2], t) ::gpu;
constraint x2>8;
constraint x0<3;
solve satisfy;