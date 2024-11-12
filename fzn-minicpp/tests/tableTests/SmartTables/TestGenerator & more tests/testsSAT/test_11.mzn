%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 83..222 : x0;
var 31..141 : x1;
var 25..122 : x2;
var 61..290 : x3;
var 18..127 : x4;
var 9..56 : x5;
var 63..137 : x6;
var 69..86 : x7;
var 20..63 : x8;
var 4..247 : x9;
var 43..96 : x10;
var 24..149 : x11;
var 49..94 : x12;
var 18..41 : x13;
var 7..74 : x14;
var 56..184 : x15;
var 20..192 : x16;
var 59..83 : x17;
var 28..223 : x18;
var 75..178 : x19;
var 87..175 : x20;
var 12..155 : x21;
var 19..186 : x22;
var 7..117 : x23;
var 17..75 : x24;
var 25..273 : x25;
var 88..283 : x26;
var 39..211 : x27;
var 80..175 : x28;
var 76..177 : x29;
var 81..284 : x30;
var 50..136 : x31;
var 65..173 : x32;
var 78..198 : x33;
var 47..268 : x34;
array [int,int] of 4..290 : t0=[|11,106,54,186,30,154,37,117,167,77,25,126,64,57,199,145,66,219,150|
50,58,56,106,27,109,27,107,186,70,26,71,122,52,95,89,50,213,122|
61,122,89,159,43,249,44,84,138,80,19,74,143,44,166,264,78,149,56|
65,49,62,130,26,61,60,92,138,69,40,145,240,37,263,262,122,51,131|
38,133,86,143,95,174,18,51,96,86,22,94,287,22,243,242,72,59,67|
73,148,51,91,115,204,55,72,144,86,40,83,76,44,193,239,80,58,210|
41,39,81,178,24,178,20,80,210,82,41,169,187,61,147,82,52,130,123|
65,127,76,118,74,150,54,135,158,78,39,69,256,20,243,224,123,210,116|
73,21,68,106,111,150,39,96,105,72,31,77,99,40,225,243,135,171,171|
35,139,51,113,60,203,47,44,195,85,26,139,222,20,201,109,103,80,60|
71,165,82,95,7,158,54,28,215,69,26,98,175,37,92,162,120,130,80|
40,97,89,133,51,153,73,89,220,71,23,153,180,36,273,250,89,105,154|
9,53,67,105,39,123,59,65,216,73,35,107,220,38,252,218,106,69,115|
21,95,94,183,49,160,21,85,152,83,23,167,235,51,171,228,59,142,201|
60,144,62,140,74,194,39,100,164,74,37,167,118,24,274,171,128,104,148|
27,22,71,118,28,188,75,29,158,82,22,165,247,48,178,196,109,134,166|
54,175,85,135,11,142,45,84,125,76,38,110,265,46,94,81,113,76,199|
67,96,50,116,69,152,32,99,136,83,31,134,130,58,127,266,99,96,109|
43,81,58,103,42,152,21,100,178,81,32,167,217,39,158,271,90,195,68|
61,51,47,192,38,48,23,118,199,70,19,61,85,60,126,178,60,183,84|
7,43,85,122,34,194,22,65,196,70,37,59,150,55,269,276,58,44,162|
32,92,89,79,89,213,38,32,154,74,35,159,144,63,161,171,69,221,154|
44,166,46,146,51,237,25,28,101,74,37,122,90,40,158,131,136,91,134|
27,59,74,138,70,52,75,38,153,78,31,69,170,36,147,166,130,214,106|
36,177,52,88,116,107,55,124,186,83,36,65,231,39,275,252,98,54,151|
37,166,86,131,60,127,61,130,118,69,37,183,77,29,251,227,77,141,45|
14,96,66,99,28,198,65,24,137,83,39,119,106,63,153,188,50,206,177|
27,145,70,89,32,105,21,65,130,76,28,115,175,63,228,236,129,214,82|
67,109,83,167,33,47,66,63,189,79,31,183,243,29,138,267,64,132,111|
62,20,91,79,28,260,46,39,155,79,39,145,152,53,98,134,64,66,207|
52,163,70,198,44,157,51,135,182,84,36,180,264,52,98,283,110,32,78|
38,108,51,147,52,93,64,62,89,69,23,158,141,61,168,253,100,210,92|
47,133,87,118,23,227,59,46,181,74,29,66,157,60,270,270,89,158,177|
56,181,95,88,71,131,25,114,155,84,22,96,181,39,262,170,117,159,70|
72,83,73,147,42,66,30,28,221,83,33,114,240,51,253,277,75,204,74|
64,70,78,180,101,170,65,29,186,75,40,125,244,50,281,171,76,221,116|
67,112,68,175,91,175,32,140,164,75,26,72,186,46,216,231,131,127,87|
50,163,52,116,57,148,32,41,131,83,20,90,74,55,241,233,125,127,39|
33,64,73,145,51,175,71,97,172,86,20,145,62,51,153,162,90,118,180|
8,186,44,144,109,174,42,68,189,78,40,87,219,32,176,180,89,125,49|
10,185,88,186,46,146,34,73,148,76,36,60,208,47,195,265,111,181,169|
63,113,51,179,11,265,61,135,142,71,34,178,129,50,263,283,128,216,125|
63,25,75,81,13,74,34,145,170,70,20,132,147,29,252,269,51,60,194|
26,147,66,162,96,220,71,58,141,71,32,171,80,33,158,131,123,195,93|
65,183,89,84,50,199,50,84,120,70,40,133,278,58,144,94,55,194,125|
59,40,59,167,17,237,32,124,205,82,22,181,186,21,166,200,133,187,155|
44,179,65,89,108,49,55,122,89,78,21,108,274,63,218,107,115,201,85|
35,132,64,122,44,117,70,102,98,75,36,163,194,23,174,96,54,188,130|
74,81,76,138,99,172,32,119,220,84,37,81,123,45,251,153,134,107,164|
69,61,78,108,44,97,36,74,88,75,38,115,199,58,105,259,135,181,191|
37,76,80,176,58,55,74,82,200,85,21,61,62,41,266,209,116,110,40|
11,63,48,193,80,266,69,127,198,83,41,110,209,38,132,170,135,181,84|
20,20,43,137,37,173,67,75,202,80,22,79,284,28,125,120,118,217,151|
54,87,71,194,28,173,20,31,107,74,34,102,219,55,129,250,58,98,97|
55,85,68,105,52,217,37,71,126,70,27,133,113,34,243,283,104,143,110|
68,186,47,79,43,94,18,95,205,86,24,104,132,48,213,113,115,123,49|]; 
array [int,int] of SmartTableOp: signs0=[|GtInt,Int,LtInt,Int,All,Int,Int,Int,LtInt,LtInt,Int,LtInt,Int,Int,Int,LtInt,All,LtInt,All|Int,All,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,GtInt,All,LtInt,Int,Int,Int,Int,Int|All,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,GtInt|Int,GtInt,Int,Int,Int,Int,GtInt,Int,Int,All,LtInt,All,LtInt,Int,All,GtInt,Int,Int,Int|All,Int,Int,Int,GtInt,All,GtInt,Int,LtInt,Int,Int,Int,All,Int,Int,Int,Int,Int,GtInt|Int,Int,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int,Int,LtInt,Int,GtInt,GtInt,Int,Int,GtInt|Int,GtInt,GtInt,Int,LtInt,Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt|Int,Int,Int,Int,GtInt,Int,Int,Int,GtInt,Int,GtInt,GtInt,Int,GtInt,GtInt,Int,Int,Int,Int|Int,GtInt,Int,Int,GtInt,Int,Int,Int,GtInt,Int,GtInt,GtInt,GtInt,Int,Int,Int,Int,Int,All|LtInt,Int,Int,GtInt,All,GtInt,All,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int,Int|Int,LtInt,Int,Int,Int,All,Int,All,Int,Int,Int,GtInt,LtInt,LtInt,Int,LtInt,Int,Int,All|GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,LtInt,Int,Int,Int,Int,LtInt,LtInt|Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,All,GtInt,Int,Int,Int,GtInt|Int,GtInt,Int,Int,Int,LtInt,Int,GtInt,Int,Int,Int,Int,All,Int,All,Int,LtInt,All,Int|All,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int,GtInt,All,Int,GtInt,LtInt,Int|Int,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,Int,LtInt,LtInt,Int,Int,Int,Int,Int|LtInt,Int,Int,LtInt,Int,All,Int,Int,All,LtInt,GtInt,LtInt,GtInt,Int,Int,All,Int,GtInt,LtInt|Int,GtInt,Int,GtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int|GtInt,Int,Int,Int,GtInt,Int,Int,Int,All,Int,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int|GtInt,Int,Int,GtInt,GtInt,All,LtInt,LtInt,Int,All,Int,Int,All,Int,All,Int,All,Int,Int|GtInt,Int,GtInt,LtInt,All,LtInt,Int,All,Int,LtInt,LtInt,Int,Int,GtInt,Int,All,Int,LtInt,GtInt|Int,All,LtInt,Int,Int,All,LtInt,Int,Int,GtInt,GtInt,All,Int,Int,GtInt,Int,Int,LtInt,LtInt|Int,GtInt,GtInt,All,Int,Int,LtInt,LtInt,Int,LtInt,LtInt,LtInt,LtInt,Int,All,GtInt,LtInt,LtInt,GtInt|LtInt,LtInt,LtInt,GtInt,Int,GtInt,GtInt,Int,Int,LtInt,LtInt,Int,GtInt,Int,Int,Int,Int,Int,GtInt|GtInt,LtInt,Int,Int,Int,Int,GtInt,Int,GtInt,Int,LtInt,Int,Int,Int,Int,Int,LtInt,GtInt,Int|Int,Int,GtInt,Int,LtInt,Int,Int,LtInt,Int,Int,LtInt,Int,All,Int,GtInt,LtInt,GtInt,All,GtInt|GtInt,All,Int,GtInt,Int,Int,GtInt,All,Int,LtInt,GtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int|Int,Int,Int,GtInt,GtInt,Int,Int,LtInt,Int,Int,GtInt,Int,GtInt,Int,GtInt,GtInt,Int,All,Int|GtInt,GtInt,GtInt,Int,Int,GtInt,LtInt,Int,Int,Int,All,Int,Int,Int,Int,Int,Int,GtInt,Int|GtInt,LtInt,GtInt,LtInt,Int,Int,All,GtInt,Int,Int,Int,All,Int,Int,GtInt,LtInt,All,GtInt,Int|Int,LtInt,Int,Int,All,Int,LtInt,GtInt,Int,Int,Int,Int,LtInt,Int,All,Int,LtInt,Int,LtInt|Int,GtInt,Int,Int,LtInt,GtInt,Int,LtInt,Int,Int,Int,All,Int,Int,Int,Int,LtInt,LtInt,LtInt|LtInt,Int,All,All,Int,Int,LtInt,LtInt,Int,Int,Int,GtInt,GtInt,GtInt,Int,LtInt,GtInt,Int,Int|Int,Int,Int,All,Int,Int,Int,LtInt,LtInt,Int,GtInt,Int,LtInt,All,All,LtInt,Int,Int,Int|All,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,All,Int,All,Int,Int|Int,Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,LtInt,All,LtInt,All|Int,GtInt,All,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int|GtInt,LtInt,Int,LtInt,Int,GtInt,Int,All,Int,Int,LtInt,Int,All,Int,LtInt,GtInt,Int,Int,GtInt|Int,Int,Int,Int,Int,Int,GtInt,LtInt,Int,LtInt,LtInt,GtInt,Int,Int,Int,Int,LtInt,Int,Int|Int,All,GtInt,LtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int|Int,Int,All,GtInt,GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,LtInt,Int|GtInt,GtInt,Int,LtInt,Int,All,GtInt,All,Int,Int,Int,Int,All,GtInt,Int,LtInt,All,LtInt,Int|Int,LtInt,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int,GtInt,GtInt,LtInt,Int,Int,GtInt|Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,GtInt,Int,LtInt|Int,Int,Int,LtInt,GtInt,Int,Int,Int,LtInt,GtInt,All,Int,GtInt,Int,LtInt,Int,Int,Int,Int|Int,LtInt,Int,Int,All,Int,Int,Int,All,GtInt,LtInt,LtInt,Int,Int,Int,Int,Int,LtInt,GtInt|Int,LtInt,Int,Int,Int,Int,LtInt,GtInt,Int,All,LtInt,GtInt,Int,Int,All,Int,Int,Int,Int|LtInt,GtInt,GtInt,GtInt,Int,GtInt,Int,GtInt,Int,GtInt,All,LtInt,Int,Int,All,All,Int,All,Int|Int,Int,LtInt,Int,Int,All,LtInt,All,Int,Int,GtInt,Int,LtInt,Int,LtInt,Int,Int,Int,Int|GtInt,LtInt,GtInt,All,LtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,All,Int|LtInt,All,Int,LtInt,LtInt,GtInt,All,Int,Int,LtInt,Int,Int,Int,All,Int,LtInt,Int,GtInt,GtInt|LtInt,LtInt,Int,GtInt,Int,All,LtInt,All,Int,Int,Int,GtInt,Int,Int,GtInt,GtInt,Int,GtInt,Int|Int,All,Int,Int,LtInt,Int,Int,Int,LtInt,Int,Int,Int,Int,All,Int,GtInt,Int,Int,LtInt|GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,Int|Int,Int,Int,Int,Int,Int,LtInt,Int,Int,LtInt,LtInt,Int,Int,GtInt,LtInt,Int,GtInt,LtInt,Int|LtInt,LtInt,Int,Int,LtInt,Int,LtInt,Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt|]; 
constraint smart_table([x14,x22,x10,x33,x23,x34,x24,x11,x0,x7,x13,x15,x3,x8,x26,x30,x31,x18,x27],t0,signs0)::uniud;
constraint not(x14=72);
constraint x22<163;
constraint not(x10=61);
constraint not(x33=101);
constraint not(x23=108);
constraint not(x34=208);
constraint x24=73;
solve satisfy;