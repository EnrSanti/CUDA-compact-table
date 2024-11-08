%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 142..459 : x0;
var 13..245 : x1;
var 110..243 : x2;
var 34..308 : x3;
var 9..70 : x4;
var 10..109 : x5;
var 9..326 : x6;
array [int,int] of 9..459 : t0=[|68,195,217,46|
95,25,185,34|
48,34,145,29|
95,96,282,68|
12,76,120,16|
47,221,58,58|
79,90,90,54|
23,27,276,25|
100,21,123,12|]; 
constraint table([x5,x1,x6,x4],t0)::uniud;
solve satisfy;