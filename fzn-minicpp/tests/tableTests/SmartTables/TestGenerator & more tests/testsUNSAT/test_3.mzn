%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 26..236 : x0;
var 31..77 : x1;
var 14..117 : x2;
var 1..149 : x3;
var 35..78 : x4;
var 81..227 : x5;
var 46..206 : x6;
var 90..115 : x7;
var 61..193 : x8;
var 69..80 : x9;
var 5..128 : x10;
var 57..237 : x11;
var 40..272 : x12;
var 25..223 : x13;
var 65..222 : x14;
var 17..265 : x15;
var 74..187 : x16;
var 9..57 : x17;
var 1..107 : x18;
var 60..83 : x19;
var 62..112 : x20;
var 89..219 : x21;
var 21..32 : x22;
var 58..247 : x23;
var 90..271 : x24;
array [int,int] of 1..272 : t0=[|74,28,60,99,65,134,136,195,26,140,101,109,63,107|
56,44,40,152,151,181,141,215,28,140,90,148,56,39|
24,52,59,99,158,182,221,174,23,183,92,106,229,123|
40,9,37,182,136,87,27,131,29,93,114,130,177,73|
80,9,72,158,225,34,194,202,21,176,95,181,86,65|
86,43,60,100,95,208,96,136,32,115,99,148,42,93|
15,9,69,96,135,125,242,158,22,141,106,154,75,72|
19,9,38,96,238,112,144,213,27,165,96,167,146,105|
89,25,66,90,159,69,216,84,32,165,110,192,29,83|
36,11,53,100,56,115,197,123,28,165,92,187,232,8|
19,41,68,155,76,215,28,161,31,117,109,121,187,100|
82,56,31,166,151,188,53,162,26,132,95,118,41,49|
35,26,53,113,45,81,187,117,32,180,111,69,197,90|
45,22,55,98,144,82,96,106,32,186,108,183,127,51|
103,48,48,74,165,62,198,216,21,124,112,94,97,103|
9,24,66,180,251,168,35,119,22,132,104,218,31,110|
72,9,58,178,132,174,256,182,30,137,108,74,157,118|
8,49,36,119,220,191,145,88,30,95,107,211,214,102|
9,45,50,188,208,184,256,83,28,159,101,193,33,125|
104,32,44,67,69,136,197,216,31,89,91,217,139,53|
59,44,44,103,225,100,178,148,25,187,113,77,214,61|
100,52,49,183,87,183,127,139,22,90,99,73,175,127|
28,52,77,172,80,52,66,110,23,83,110,170,66,85|
52,57,58,135,216,80,254,186,22,184,111,180,177,5|
94,46,36,75,214,55,148,223,24,93,102,149,121,90|
11,36,57,147,56,203,177,104,23,123,94,169,145,68|
5,18,56,169,67,65,126,202,32,132,92,85,113,113|
62,44,60,123,162,70,228,180,24,103,108,195,141,99|
95,51,77,166,61,171,215,128,22,169,114,159,179,9|
86,24,52,186,164,187,231,208,31,149,113,184,113,13|
47,20,66,121,178,176,124,89,22,138,94,187,158,128|
31,38,46,153,131,170,142,95,25,168,104,112,181,47|
59,35,37,66,173,198,254,135,27,182,102,78,213,60|
45,14,33,65,142,103,111,168,22,137,101,197,101,11|
104,53,38,128,178,81,188,97,23,79,109,170,156,127|
76,52,31,83,228,82,23,202,24,94,113,177,100,67|
30,31,41,171,218,220,226,116,24,122,95,161,162,60|
38,56,34,114,106,173,61,173,22,102,111,137,141,53|
44,34,72,102,206,127,182,142,32,182,90,137,218,76|
92,17,36,111,152,28,111,213,28,172,96,100,122,126|
8,23,76,173,65,87,139,126,30,78,101,149,28,17|
93,31,70,164,90,222,70,107,29,108,94,201,40,5|
51,28,45,69,103,176,159,157,31,112,98,158,31,91|
78,31,68,137,114,87,159,98,31,96,90,95,61,31|
92,24,66,177,142,118,34,168,25,138,102,178,136,62|
91,50,46,129,152,156,148,224,29,92,110,103,61,79|
41,36,72,75,241,172,172,159,25,148,110,104,184,11|
43,23,76,103,175,159,214,179,30,136,104,189,46,52|
69,42,77,151,140,76,260,107,25,88,101,109,87,111|
105,43,75,108,183,216,175,172,27,187,105,151,229,107|
38,32,71,146,238,86,200,132,32,125,95,130,187,46|
83,11,40,143,104,152,88,144,27,143,103,166,126,6|
76,50,50,146,171,113,111,117,28,95,106,121,211,18|
86,26,40,130,196,115,148,200,26,174,111,133,97,28|
105,15,33,154,251,205,133,87,29,103,94,175,136,5|
46,21,53,98,227,90,192,138,32,133,103,99,213,18|
92,46,56,193,138,116,227,168,21,160,112,152,80,17|
16,44,42,84,123,165,263,157,21,94,91,131,141,32|
52,34,42,140,123,110,259,95,32,87,111,126,53,113|
42,15,59,147,106,80,158,138,23,107,109,205,135,39|
32,32,52,128,98,66,233,137,24,103,111,106,190,61|
79,12,68,162,87,55,242,99,32,108,112,188,63,74|
25,52,48,143,216,176,51,224,25,158,104,119,85,27|]; 
array [int,int] of SmartTableOp: signs0=[|Int,Int,LtInt,LtInt,Int,Int,Int,Int,LtInt,Int,Int,LtInt,GtInt,GtInt|Int,Int,GtInt,Int,GtInt,GtInt,LtInt,Int,Int,GtInt,GtInt,LtInt,Int,Int|All,Int,Int,All,All,GtInt,LtInt,Int,GtInt,Int,All,Int,LtInt,GtInt|Int,Int,LtInt,Int,All,Int,LtInt,Int,LtInt,Int,GtInt,Int,GtInt,Int|GtInt,LtInt,GtInt,GtInt,LtInt,Int,Int,Int,GtInt,Int,LtInt,Int,Int,Int|GtInt,Int,LtInt,Int,All,Int,LtInt,All,GtInt,GtInt,Int,Int,LtInt,Int|Int,Int,LtInt,Int,Int,All,LtInt,Int,LtInt,Int,Int,GtInt,LtInt,GtInt|Int,GtInt,Int,Int,Int,Int,Int,Int,Int,All,All,Int,Int,Int|All,Int,All,All,Int,LtInt,LtInt,Int,Int,LtInt,GtInt,GtInt,Int,Int|Int,Int,Int,Int,Int,All,All,Int,Int,Int,GtInt,Int,Int,GtInt|Int,GtInt,LtInt,All,Int,GtInt,Int,Int,Int,All,Int,LtInt,Int,Int|Int,LtInt,Int,GtInt,Int,GtInt,Int,Int,Int,Int,Int,LtInt,LtInt,LtInt|Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,Int|Int,Int,GtInt,Int,LtInt,All,LtInt,GtInt,Int,LtInt,Int,LtInt,GtInt,Int|Int,All,Int,LtInt,Int,Int,All,GtInt,Int,All,All,Int,All,All|Int,Int,GtInt,Int,LtInt,Int,Int,Int,Int,Int,All,GtInt,Int,Int|Int,GtInt,All,Int,Int,Int,All,Int,Int,Int,Int,Int,LtInt,Int|Int,Int,Int,Int,Int,LtInt,Int,GtInt,All,All,Int,Int,Int,Int|LtInt,Int,LtInt,Int,Int,Int,LtInt,LtInt,Int,Int,LtInt,All,LtInt,Int|GtInt,LtInt,GtInt,Int,LtInt,Int,Int,Int,Int,LtInt,Int,Int,All,LtInt|Int,Int,Int,LtInt,GtInt,GtInt,All,Int,LtInt,LtInt,All,All,Int,Int|Int,Int,Int,All,GtInt,All,Int,Int,LtInt,Int,LtInt,Int,GtInt,Int|LtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,LtInt,LtInt,Int,All,Int|Int,Int,Int,Int,Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int,GtInt|Int,Int,Int,Int,Int,GtInt,Int,Int,Int,GtInt,LtInt,GtInt,Int,Int|Int,GtInt,LtInt,LtInt,Int,Int,GtInt,Int,Int,Int,GtInt,LtInt,All,Int|Int,Int,Int,Int,Int,Int,All,Int,LtInt,Int,Int,Int,LtInt,All|Int,Int,LtInt,Int,Int,GtInt,Int,LtInt,LtInt,GtInt,Int,Int,Int,All|Int,Int,LtInt,Int,Int,Int,All,Int,GtInt,All,Int,Int,All,GtInt|Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,All,All,All,GtInt,Int|Int,Int,Int,Int,LtInt,Int,GtInt,All,Int,Int,Int,GtInt,Int,All|Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt|Int,Int,Int,Int,Int,All,Int,Int,All,Int,All,All,Int,Int|Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,LtInt|GtInt,LtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,Int,Int,Int,Int|LtInt,Int,LtInt,LtInt,Int,GtInt,LtInt,Int,Int,All,Int,LtInt,GtInt,Int|LtInt,Int,LtInt,Int,GtInt,LtInt,Int,GtInt,Int,All,GtInt,LtInt,Int,Int|GtInt,Int,Int,All,LtInt,LtInt,Int,Int,Int,Int,All,LtInt,Int,GtInt|Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,GtInt,LtInt|All,Int,GtInt,GtInt,Int,Int,Int,Int,GtInt,LtInt,Int,GtInt,Int,Int|GtInt,All,GtInt,LtInt,LtInt,Int,Int,Int,LtInt,GtInt,All,LtInt,Int,Int|LtInt,All,GtInt,LtInt,GtInt,All,Int,LtInt,Int,Int,Int,Int,LtInt,LtInt|LtInt,All,LtInt,Int,LtInt,GtInt,All,Int,Int,Int,All,Int,Int,All|Int,Int,Int,GtInt,GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,Int|Int,GtInt,GtInt,LtInt,Int,All,Int,Int,Int,GtInt,LtInt,Int,Int,Int|Int,All,Int,Int,GtInt,Int,LtInt,LtInt,Int,Int,GtInt,Int,LtInt,Int|Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,All,LtInt,LtInt,Int|Int,Int,GtInt,Int,Int,All,Int,All,All,Int,GtInt,Int,Int,Int|LtInt,Int,Int,Int,All,Int,Int,Int,All,GtInt,Int,Int,Int,Int|GtInt,LtInt,Int,Int,LtInt,Int,LtInt,GtInt,Int,All,LtInt,All,All,Int|LtInt,LtInt,Int,Int,Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int,Int|LtInt,LtInt,All,Int,All,Int,Int,Int,LtInt,GtInt,Int,GtInt,Int,All|Int,LtInt,LtInt,Int,GtInt,Int,LtInt,Int,GtInt,LtInt,LtInt,All,All,Int|All,GtInt,LtInt,Int,LtInt,LtInt,All,Int,Int,Int,Int,Int,Int,LtInt|All,Int,LtInt,GtInt,GtInt,Int,GtInt,GtInt,All,LtInt,Int,Int,LtInt,Int|Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int|Int,Int,LtInt,Int,Int,All,Int,All,LtInt,GtInt,Int,LtInt,Int,LtInt|LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,All,Int|Int,Int,LtInt,Int,Int,All,Int,Int,Int,Int,GtInt,LtInt,Int,Int|Int,GtInt,Int,All,Int,GtInt,All,Int,Int,Int,GtInt,Int,GtInt,Int|Int,Int,GtInt,LtInt,Int,LtInt,All,Int,LtInt,All,Int,Int,GtInt,GtInt|Int,All,GtInt,Int,LtInt,LtInt,Int,GtInt,Int,Int,Int,Int,Int,Int|Int,GtInt,Int,Int,All,Int,GtInt,Int,All,LtInt,Int,Int,Int,Int|]; 
constraint smart_table([x18,x17,x1,x8,x12,x13,x15,x5,x22,x16,x7,x14,x0,x10],t0,signs0)::uniud;
constraint not(x18=47);
constraint x17=19;
constraint x1=73;
constraint x8=68;
solve satisfy;