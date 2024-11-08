%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 43..119 : x0;
var 33..97 : x1;
var 50..148 : x2;
var 47..107 : x3;
array [int,int] of 33..148 : t0=[|68|
116|
112|
62|
97|
74|
44|
62|
95|
81|
102|
91|
109|]; 
constraint table([x0],t0)::uniud;
solve satisfy;