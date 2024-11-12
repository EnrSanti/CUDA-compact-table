%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 4..31 : x0;
var 5..67 : x1;
var 11..49 : x2;
var 12..23 : x3;
var 35..94 : x4;
var 14..49 : x5;
var 7..41 : x6;
var 15..76 : x7;
var 18..82 : x8;
var 29..54 : x9;
var 30..88 : x10;
var 15..59 : x11;
var 39..114 : x12;
var 23..82 : x13;
var 28..71 : x14;
var 37..105 : x15;
var 22..65 : x16;
var 33..71 : x17;
var 12..61 : x18;
array [int,int] of 4..114 : t0=[|84,23,52,16,54,53,45,40,75|
82,29,46,38,47,19,31,6,50|
64,10,30,24,45,49,51,57,72|
78,24,44,32,36,36,39,56,82|
58,18,16,7,49,45,51,6,72|
51,8,18,27,76,51,27,34,24|
81,8,53,37,37,43,28,50,22|
63,4,17,12,54,37,47,30,30|
54,17,16,33,71,46,27,8,65|
49,29,33,9,24,43,39,63,43|
45,4,15,31,32,20,51,48,51|
91,19,24,21,62,61,50,12,26|
92,6,18,9,25,29,34,22,69|
43,24,42,16,46,48,62,50,49|
45,9,48,31,77,31,55,38,50|
88,22,48,8,51,21,23,52,78|
48,17,18,35,39,32,49,43,23|
42,18,45,12,49,44,40,64,53|
59,21,49,36,34,34,46,30,49|
60,14,47,8,72,60,65,25,19|
37,30,55,39,68,50,25,19,38|
63,9,27,12,82,34,52,51,32|
39,20,25,40,72,25,57,19,63|
73,29,51,20,49,33,33,15,77|
82,30,44,32,40,16,58,54,39|
44,19,55,27,78,23,33,45,27|
37,23,22,39,53,43,48,37,47|
57,4,59,20,35,59,45,51,77|
50,13,36,10,38,12,34,43,79|
91,22,25,22,42,43,31,63,55|
71,4,24,18,73,40,44,43,77|
70,7,58,31,42,54,35,66,70|
48,7,59,8,23,27,34,33,33|
77,14,49,22,26,50,46,53,34|
41,25,17,38,33,58,34,38,82|
48,8,38,32,33,39,31,52,27|
64,24,37,37,74,56,41,16,58|
90,28,30,18,73,32,24,30,28|
50,15,48,14,81,33,24,64,27|
43,24,54,19,59,59,64,45,29|
91,26,25,12,70,29,45,60,51|
79,28,40,32,36,37,36,48,65|
55,11,36,9,72,40,30,37,69|
41,16,59,7,39,30,48,65,39|
66,24,35,28,31,26,33,22,69|
92,24,46,17,45,40,26,16,46|
49,10,47,35,31,12,63,29,47|
91,25,29,32,57,13,45,64,52|
52,25,21,12,67,59,64,49,73|
38,27,46,20,37,15,29,37,58|
68,23,15,33,53,48,27,51,54|
39,19,56,33,78,58,31,26,20|
88,6,50,9,30,31,36,26,19|
80,14,50,13,52,49,41,26,45|
87,14,54,8,74,40,43,62,64|
71,22,32,25,52,46,64,17,55|
63,23,58,14,43,13,36,26,64|
49,15,49,38,62,36,26,51,24|
52,26,27,40,66,45,24,64,78|
55,31,46,15,49,33,60,31,56|
37,30,29,34,25,61,30,49,34|
70,4,39,29,36,40,32,10,35|
59,7,55,31,35,58,32,26,18|
48,21,30,28,41,21,47,57,46|
42,31,26,31,36,45,54,33,34|
64,13,21,12,65,18,64,41,51|
67,23,51,17,79,27,45,58,40|
84,17,43,37,30,43,37,12,34|
48,23,37,7,53,37,42,33,44|
43,24,17,23,58,22,28,9,57|
60,25,57,26,64,46,49,16,21|
79,6,21,20,24,26,64,36,32|
57,12,24,40,48,53,32,29,24|
45,18,17,33,25,34,23,38,35|
58,24,43,25,40,38,47,8,46|
76,13,47,31,66,44,63,12,82|]; 
array [int,int] of SmartTableOp: signs0=[|LtInt,Int,Int,Int,GtInt,Int,Int,Int,LtInt|All,Int,Int,All,LtInt,Int,Int,Int,Int|Int,Int,Int,Int,LtInt,GtInt,Int,Int,Int|Int,Int,Int,Int,All,LtInt,GtInt,GtInt,All|Int,Int,All,All,Int,Int,Int,GtInt,Int|Int,GtInt,Int,Int,Int,Int,GtInt,GtInt,LtInt|LtInt,Int,Int,Int,Int,Int,Int,LtInt,Int|Int,LtInt,GtInt,LtInt,All,Int,Int,LtInt,All|Int,Int,Int,Int,Int,Int,Int,LtInt,Int|Int,Int,Int,Int,GtInt,All,Int,LtInt,Int|Int,Int,GtInt,LtInt,Int,GtInt,LtInt,Int,Int|Int,Int,Int,Int,GtInt,LtInt,Int,Int,LtInt|LtInt,Int,All,GtInt,GtInt,Int,GtInt,Int,LtInt|LtInt,Int,Int,All,Int,Int,Int,LtInt,Int|Int,LtInt,All,Int,All,Int,Int,Int,LtInt|GtInt,LtInt,Int,Int,GtInt,Int,All,GtInt,LtInt|Int,All,Int,LtInt,GtInt,Int,LtInt,GtInt,LtInt|Int,Int,Int,LtInt,Int,All,Int,Int,GtInt|All,Int,Int,Int,All,Int,Int,Int,GtInt|LtInt,Int,Int,Int,Int,LtInt,GtInt,Int,All|Int,Int,GtInt,GtInt,Int,Int,GtInt,Int,Int|LtInt,GtInt,Int,GtInt,Int,Int,Int,Int,LtInt|LtInt,Int,Int,All,Int,Int,Int,Int,Int|LtInt,Int,Int,Int,GtInt,Int,LtInt,LtInt,Int|Int,Int,Int,Int,GtInt,Int,LtInt,LtInt,LtInt|Int,LtInt,Int,GtInt,Int,Int,Int,All,Int|Int,Int,Int,All,Int,Int,Int,LtInt,GtInt|Int,All,Int,Int,Int,LtInt,LtInt,LtInt,GtInt|All,Int,Int,GtInt,Int,Int,Int,Int,Int|LtInt,Int,LtInt,GtInt,LtInt,Int,LtInt,LtInt,LtInt|LtInt,Int,LtInt,Int,Int,Int,GtInt,Int,Int|GtInt,All,GtInt,Int,Int,All,LtInt,Int,Int|Int,Int,Int,Int,Int,GtInt,Int,Int,All|Int,LtInt,Int,GtInt,Int,All,All,LtInt,Int|Int,LtInt,Int,All,GtInt,All,All,Int,Int|Int,Int,Int,GtInt,Int,All,Int,Int,GtInt|Int,GtInt,Int,All,Int,All,Int,Int,Int|Int,GtInt,Int,Int,Int,Int,All,Int,GtInt|LtInt,All,Int,Int,Int,LtInt,LtInt,Int,Int|GtInt,Int,Int,Int,All,Int,Int,GtInt,Int|Int,All,LtInt,GtInt,LtInt,Int,Int,Int,Int|Int,LtInt,Int,All,Int,Int,Int,Int,Int|GtInt,Int,Int,Int,GtInt,Int,Int,Int,Int|Int,Int,Int,LtInt,Int,Int,Int,Int,Int|Int,GtInt,GtInt,Int,GtInt,LtInt,Int,GtInt,Int|LtInt,Int,Int,LtInt,GtInt,LtInt,Int,All,Int|All,All,GtInt,LtInt,LtInt,Int,Int,Int,Int|Int,GtInt,GtInt,LtInt,All,Int,Int,Int,LtInt|Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,LtInt,LtInt,Int,LtInt,Int,Int,Int,Int|Int,Int,GtInt,All,GtInt,All,Int,Int,GtInt|All,All,Int,Int,GtInt,LtInt,Int,Int,LtInt|Int,Int,GtInt,Int,Int,Int,Int,Int,Int|GtInt,Int,All,LtInt,Int,LtInt,Int,GtInt,LtInt|LtInt,Int,LtInt,Int,Int,All,Int,Int,Int|Int,GtInt,Int,All,Int,All,LtInt,GtInt,Int|Int,LtInt,Int,Int,Int,Int,Int,All,LtInt|LtInt,Int,Int,Int,Int,Int,GtInt,Int,Int|Int,Int,All,Int,GtInt,All,Int,Int,Int|Int,LtInt,LtInt,LtInt,Int,LtInt,All,GtInt,Int|Int,GtInt,All,LtInt,Int,Int,GtInt,Int,LtInt|Int,Int,Int,Int,Int,Int,Int,Int,LtInt|Int,Int,Int,Int,Int,LtInt,Int,Int,Int|LtInt,Int,Int,Int,Int,LtInt,All,Int,Int|Int,GtInt,LtInt,LtInt,All,Int,Int,LtInt,Int|Int,Int,All,Int,Int,Int,Int,All,All|Int,Int,Int,Int,Int,Int,LtInt,All,GtInt|Int,Int,GtInt,GtInt,Int,LtInt,LtInt,Int,LtInt|Int,Int,Int,LtInt,Int,Int,Int,Int,Int|Int,LtInt,All,Int,LtInt,Int,GtInt,GtInt,GtInt|Int,Int,All,Int,Int,Int,Int,Int,Int|Int,GtInt,All,Int,Int,Int,Int,All,Int|Int,Int,All,GtInt,Int,GtInt,Int,All,GtInt|LtInt,GtInt,LtInt,Int,All,All,LtInt,All,GtInt|GtInt,Int,Int,All,GtInt,All,Int,Int,Int|GtInt,Int,Int,All,GtInt,Int,LtInt,Int,Int|]; 
constraint smart_table([x4,x0,x11,x6,x13,x18,x16,x1,x8],t0,signs0)::gpu;
constraint not(x4=75);
constraint not(x0=24);
constraint not(x11=39);
constraint not(x6=25);
constraint not(x13=69);
constraint not(x18=47);
constraint not(x16=55);
constraint x1=47;
constraint not(x8=43);
solve satisfy;