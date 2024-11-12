%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 20..138 : x0;
var 64..174 : x1;
var 22..126 : x2;
var 86..322 : x3;
var 0..252 : x4;
var 75..109 : x5;
var 15..45 : x6;
var 7..43 : x7;
var 68..82 : x8;
var 9..174 : x9;
var 53..274 : x10;
var 84..110 : x11;
array [int,int] of 7..322 : t0=[|76,106,115,112,33,30|
41,103,96,74,44,29|
49,94,175,74,34,13|
138,90,181,22,22,34|
124,85,216,37,45,27|
75,90,60,85,38,30|
155,96,223,83,16,24|
135,105,170,92,24,37|
91,90,153,109,34,31|
146,105,73,92,32,9|
114,105,195,62,27,13|
116,93,192,78,19,16|
100,110,197,114,27,35|
111,106,216,109,19,13|
114,89,216,119,24,7|
101,84,158,66,34,37|
34,99,210,126,41,7|
28,103,232,115,41,8|
71,98,199,75,44,17|
100,95,184,59,15,42|
81,89,89,42,15,11|
15,109,223,79,31,30|
143,109,95,58,38,30|
82,92,200,67,15,36|
110,92,59,110,16,28|
120,110,231,29,19,18|
39,92,165,63,34,35|
120,93,169,34,18,15|
97,84,117,44,42,13|
23,100,174,107,24,10|
57,106,228,73,44,33|
157,103,107,91,43,33|
135,96,64,75,15,24|
154,101,101,114,18,11|
35,106,118,99,23,39|
45,97,172,107,29,23|
96,103,135,70,20,41|
64,102,98,52,36,11|
145,89,98,31,30,33|
78,100,67,118,31,36|
52,87,112,50,21,8|
92,110,48,90,27,20|
60,101,99,88,39,28|]; 
array [int,int] of SmartTableOp: signs0=[|Int,Int,Int,Int,Int,Int|GtInt,Int,LtInt,LtInt,GtInt,GtInt|Int,Int,Int,GtInt,Int,GtInt|GtInt,Int,GtInt,Int,Int,Int|Int,GtInt,LtInt,LtInt,Int,All|Int,LtInt,All,Int,LtInt,Int|Int,LtInt,GtInt,LtInt,GtInt,Int|Int,Int,Int,Int,GtInt,Int|LtInt,GtInt,Int,Int,All,Int|Int,Int,All,All,Int,Int|LtInt,Int,Int,GtInt,Int,Int|Int,Int,LtInt,GtInt,Int,GtInt|Int,Int,Int,Int,Int,Int|Int,Int,LtInt,All,Int,Int|Int,GtInt,Int,Int,LtInt,LtInt|Int,Int,Int,Int,Int,LtInt|All,LtInt,Int,Int,LtInt,LtInt|Int,LtInt,Int,Int,Int,Int|LtInt,Int,Int,Int,LtInt,All|Int,Int,All,Int,Int,Int|All,LtInt,Int,Int,Int,Int|Int,Int,Int,GtInt,LtInt,LtInt|Int,Int,Int,Int,GtInt,LtInt|Int,Int,Int,Int,Int,LtInt|Int,GtInt,Int,LtInt,All,GtInt|Int,LtInt,Int,Int,Int,Int|All,All,Int,Int,Int,Int|Int,Int,LtInt,Int,Int,Int|Int,All,Int,LtInt,Int,Int|Int,Int,Int,Int,Int,Int|Int,Int,Int,Int,LtInt,Int|LtInt,Int,Int,GtInt,Int,GtInt|Int,Int,Int,Int,Int,Int|Int,GtInt,Int,LtInt,Int,GtInt|Int,GtInt,GtInt,GtInt,Int,LtInt|Int,Int,LtInt,LtInt,LtInt,Int|Int,Int,Int,Int,Int,All|Int,All,LtInt,GtInt,Int,Int|Int,All,GtInt,LtInt,Int,GtInt|Int,Int,Int,Int,All,Int|LtInt,GtInt,Int,All,LtInt,LtInt|Int,Int,Int,GtInt,Int,GtInt|All,Int,Int,Int,All,Int|]; 
constraint smart_table([x9,x11,x4,x2,x6,x7],t0,signs0)::uniud;
constraint not(x9=139);
constraint x11=107;
constraint x4<87;
constraint x2>91;
solve satisfy;