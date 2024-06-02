include "table.mzn";
include "minicpp.mzn";

var 5..6 : x0;
var 30..50 : x1;
var 5..50 : x2;

constraint x0>5;
array [int,int] of 1..50: t = [|6,30,8|6,32,9|6,32,9|6,31,9|6,36,9|6,36,40|];

constraint table([x0,x1,x2], t) ::uniud;
constraint x1>35;
constraint x2>36;
constraint x2<48;
solve satisfy;