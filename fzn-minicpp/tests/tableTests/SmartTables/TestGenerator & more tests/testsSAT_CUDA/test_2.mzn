%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 15..29 : x0;
var 22..71 : x1;
var 5..22 : x2;
var 25..39 : x3;
var 20..41 : x4;
var 2..12 : x5;
var 34..65 : x6;
var 14..40 : x7;
var 7..64 : x8;
var 29..72 : x9;
var 14..79 : x10;
array [int,int] of 2..79 : t0=[|31,42,8,26,70,52,34|
35,56,6,31,62,42,33|
70,51,11,30,52,62,40|
35,62,18,39,52,52,24|
49,22,6,31,55,56,30|
21,50,21,29,37,49,19|
58,9,8,33,70,41,40|
36,47,15,30,54,60,35|
78,46,7,33,61,35,34|
43,64,18,31,37,41,31|
52,35,6,37,30,57,24|
37,31,10,34,72,48,32|
62,38,15,30,36,40,16|
36,39,16,26,46,56,15|
69,29,9,34,49,36,18|
39,11,8,25,39,52,23|
16,11,15,29,44,61,18|
70,20,11,27,64,58,26|
61,18,8,32,41,51,37|
67,45,12,28,49,56,23|
55,58,7,35,43,57,32|
35,12,20,27,35,37,22|
57,47,10,29,35,43,18|
28,8,7,36,59,35,22|
56,60,7,34,63,59,39|
50,54,17,38,57,52,19|
17,26,11,34,58,56,31|
61,7,18,29,31,57,18|
50,54,20,37,57,49,27|
62,34,5,34,40,54,19|
69,38,19,28,47,63,22|
46,45,17,28,52,39,39|
41,40,15,34,55,61,32|
31,10,20,27,68,47,33|
75,61,7,28,45,41,26|
37,50,9,39,43,35,32|
68,30,10,39,41,52,38|
19,21,13,28,58,58,17|
65,48,15,38,33,51,22|
65,24,8,37,46,62,28|
70,59,7,25,69,53,35|
52,24,12,32,58,64,33|]; 
array [int,int] of SmartTableOp: signs0=[|LtInt,Int,GtInt,GtInt,LtInt,Int,Int|Int,LtInt,LtInt,Int,GtInt,Int,Int|GtInt,Int,LtInt,Int,LtInt,Int,Int|Int,Int,GtInt,Int,All,Int,GtInt|GtInt,LtInt,GtInt,LtInt,Int,Int,LtInt|Int,Int,Int,Int,Int,Int,Int|Int,LtInt,LtInt,Int,LtInt,Int,GtInt|LtInt,LtInt,All,Int,Int,LtInt,Int|Int,Int,All,GtInt,GtInt,Int,GtInt|Int,Int,Int,Int,Int,Int,All|Int,GtInt,Int,LtInt,Int,Int,Int|Int,Int,Int,Int,Int,Int,GtInt|Int,GtInt,Int,Int,Int,LtInt,LtInt|Int,All,Int,GtInt,Int,Int,Int|GtInt,LtInt,Int,All,Int,Int,Int|Int,All,Int,LtInt,Int,All,GtInt|Int,Int,GtInt,Int,Int,Int,Int|Int,Int,GtInt,LtInt,Int,LtInt,Int|Int,Int,Int,Int,LtInt,Int,Int|Int,All,LtInt,LtInt,Int,Int,Int|Int,Int,GtInt,Int,Int,LtInt,Int|Int,All,Int,Int,LtInt,Int,Int|LtInt,LtInt,Int,Int,LtInt,Int,LtInt|Int,Int,Int,Int,Int,Int,Int|All,Int,GtInt,GtInt,Int,LtInt,All|GtInt,GtInt,Int,Int,Int,LtInt,Int|LtInt,Int,GtInt,All,Int,Int,GtInt|Int,Int,Int,GtInt,Int,Int,GtInt|Int,Int,GtInt,LtInt,Int,Int,LtInt|GtInt,Int,Int,All,Int,GtInt,Int|Int,LtInt,GtInt,LtInt,GtInt,Int,Int|Int,GtInt,Int,Int,Int,GtInt,Int|LtInt,Int,LtInt,All,LtInt,Int,Int|Int,LtInt,Int,Int,All,Int,Int|Int,LtInt,Int,GtInt,LtInt,All,Int|Int,LtInt,All,All,GtInt,Int,Int|GtInt,LtInt,Int,Int,Int,Int,LtInt|Int,Int,Int,LtInt,LtInt,Int,LtInt|Int,Int,Int,Int,Int,LtInt,Int|All,LtInt,LtInt,GtInt,Int,Int,Int|Int,Int,LtInt,Int,LtInt,Int,Int|Int,GtInt,LtInt,Int,Int,Int,Int|]; 
constraint smart_table([x10,x8,x2,x3,x9,x6,x7],t0,signs0)::gpu;
constraint not(x10=31);
constraint not(x8=24);
constraint x2>14;
constraint not(x3=37);
constraint not(x9=55);
constraint not(x6=61);
constraint x7>15;
solve satisfy;