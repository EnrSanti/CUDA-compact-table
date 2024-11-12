%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 9..65 : x0;
var 12..145 : x1;
var 2..198 : x2;
var 36..218 : x3;
var 39..131 : x4;
var 29..62 : x5;
var 52..129 : x6;
var 12..98 : x7;
var 20..172 : x8;
var 56..155 : x9;
var 46..100 : x10;
var 1..83 : x11;
var 52..206 : x12;
var 27..115 : x13;
var 29..211 : x14;
var 38..217 : x15;
var 30..93 : x16;
var 3..37 : x17;
var 37..57 : x18;
var 53..81 : x19;
var 34..168 : x20;
var 28..139 : x21;
var 36..153 : x22;
var 60..116 : x23;
var 6..205 : x24;
var 59..197 : x25;
var 28..222 : x26;
var 59..110 : x27;
var 54..150 : x28;
var 31..156 : x29;
var 27..170 : x30;
var 29..68 : x31;
var 46..138 : x32;
var 16..146 : x33;
var 45..105 : x34;
var 38..205 : x35;
var 6..100 : x36;
var 54..158 : x37;
var 21..84 : x38;
var 56..249 : x39;
var 41..122 : x40;
var 33..192 : x41;
var 1..176 : x42;
var 13..71 : x43;
var 1..92 : x44;
array [int,int] of 1..249 : t0=[|60,131,119,70,122,66,39,60,144,108,123,62,66,42,75,105,120,52,125,12,77,42,28|
67,187,109,81,103,24,42,83,162,175,46,49,113,40,35,179,60,75,119,65,110,39,151|
74,128,89,69,79,65,41,45,184,99,24,64,79,37,30,134,47,84,128,30,101,38,167|
55,168,111,61,121,45,89,97,229,164,139,64,90,36,66,189,54,64,74,90,123,48,157|
33,45,75,71,88,83,53,11,215,43,42,37,95,65,31,113,19,48,80,29,143,62,85|
25,191,122,70,62,69,130,80,108,80,126,66,103,49,42,72,76,70,97,78,80,47,121|
44,148,66,69,149,27,108,94,225,25,99,15,68,34,107,120,48,89,60,51,61,55,104|
49,133,97,73,167,55,73,96,235,7,36,35,102,47,110,157,50,70,56,50,79,46,100|
42,73,144,79,105,50,127,46,159,119,95,42,64,68,113,167,106,53,113,89,75,57,27|
59,185,123,60,104,78,41,61,142,58,69,32,82,35,78,148,57,48,98,59,109,29,40|
11,105,72,74,90,55,110,7,202,29,17,59,109,52,29,191,22,51,78,72,126,44,94|
50,92,89,66,76,79,107,36,189,123,89,38,61,33,115,158,18,47,83,87,112,31,98|
29,88,79,73,141,84,85,47,193,56,70,14,81,66,30,179,84,88,111,72,66,32,187|
19,60,97,63,28,63,58,91,128,136,49,46,75,51,92,64,23,63,109,80,132,59,185|
83,169,101,79,100,72,102,27,130,73,77,32,97,52,103,187,144,55,81,37,142,48,110|
61,117,124,68,141,21,43,30,90,79,81,32,85,32,30,97,57,84,106,17,77,35,96|
22,60,145,67,136,66,82,41,198,24,117,51,81,41,105,153,65,62,106,89,92,46,125|
67,204,152,56,58,75,61,74,190,170,30,36,91,52,107,192,92,63,115,75,94,51,173|
53,144,86,80,78,47,102,75,119,55,58,53,72,46,84,71,41,88,92,28,95,57,39|
88,193,130,66,162,80,72,40,85,127,26,36,75,34,100,133,63,81,109,34,74,44,162|
24,71,65,78,48,84,70,85,196,50,72,46,62,58,113,93,53,65,54,17,117,45,96|
30,74,144,62,46,46,129,58,132,24,58,44,107,55,41,120,93,68,97,88,142,50,66|
15,95,79,77,61,30,75,25,204,55,84,46,61,40,88,85,145,83,120,27,139,42,144|
80,52,155,76,168,64,54,8,160,171,86,60,88,33,59,153,138,79,101,92,93,61,76|
7,169,76,55,100,66,67,19,120,51,131,65,113,40,69,167,128,93,104,29,63,62,57|
60,117,146,53,153,26,122,98,56,111,88,24,76,57,87,117,70,71,59,46,144,55,117|
91,185,110,59,50,29,60,55,138,24,61,25,67,63,70,179,40,52,77,70,123,51,17|
66,174,156,81,75,21,71,61,230,133,21,45,107,48,45,86,122,82,73,19,143,39,102|
80,113,93,74,36,26,75,64,73,158,59,38,116,32,89,100,108,62,76,28,151,42,21|
34,185,129,58,108,70,101,28,101,73,83,56,69,48,66,86,102,100,64,53,139,34,35|
18,62,115,56,92,25,109,60,165,3,62,57,66,58,105,136,132,62,59,98,79,43,41|
91,47,57,74,52,43,74,17,184,6,46,23,83,49,101,148,131,82,105,54,103,30,32|
56,118,106,74,44,68,48,92,197,120,103,41,63,44,60,74,128,76,80,96,59,45,147|
15,176,152,73,33,46,42,20,232,102,110,64,112,54,55,158,48,94,118,40,111,52,47|
72,123,109,68,20,78,104,67,128,71,19,39,110,58,90,141,85,100,109,14,69,44,128|
72,139,127,53,142,51,67,79,73,15,117,44,66,29,47,184,25,82,100,53,70,58,110|
51,42,56,65,59,44,123,17,63,67,61,22,76,49,34,174,61,87,119,44,70,43,53|
6,68,140,77,83,30,108,10,183,23,38,61,109,64,63,112,18,83,52,61,78,62,89|
2,73,126,76,82,68,95,71,226,72,28,15,66,58,107,76,89,100,109,54,82,53,121|
1,162,157,68,89,43,60,39,206,119,45,52,79,54,45,146,54,61,109,68,78,39,196|
46,86,128,59,165,46,58,24,216,45,28,43,74,48,45,197,133,64,76,70,139,48,122|
25,157,57,53,119,59,53,99,56,62,29,21,89,48,73,111,43,51,67,85,90,58,99|
18,47,128,72,139,66,121,89,229,114,144,52,60,41,28,144,144,54,113,95,57,58,162|
41,161,56,76,71,63,43,95,62,25,92,41,82,46,105,60,47,54,82,50,130,58,110|
49,149,111,77,131,82,65,72,135,111,130,56,93,29,105,185,96,73,87,58,133,40,175|
24,194,66,72,93,22,90,44,206,88,57,58,76,63,61,102,45,70,91,44,97,29,50|
74,193,142,69,99,41,94,14,127,122,22,70,61,50,49,117,122,69,73,54,126,36,86|
42,59,79,71,57,38,85,51,69,144,101,39,92,46,105,111,101,90,118,40,93,31,153|]; 
array [int,int] of SmartTableOp: signs0=[|Int,Int,Int,Int,All,Int,LtInt,LtInt,LtInt,All,Int,Int,GtInt,Int,All,Int,Int,Int,LtInt,Int,GtInt,GtInt,Int|LtInt,LtInt,Int,Int,LtInt,LtInt,All,All,Int,GtInt,LtInt,GtInt,All,Int,Int,GtInt,LtInt,Int,GtInt,All,Int,Int,Int|GtInt,Int,Int,Int,Int,GtInt,GtInt,LtInt,Int,Int,All,Int,GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int|Int,Int,GtInt,Int,All,Int,Int,GtInt,LtInt,Int,LtInt,All,All,Int,All,Int,Int,All,Int,Int,Int,Int,Int|Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,All,GtInt,Int,Int,LtInt,Int,LtInt,Int,Int,All,All,Int,LtInt,LtInt|Int,GtInt,Int,LtInt,LtInt,All,GtInt,Int,GtInt,GtInt,Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,GtInt,Int,GtInt,LtInt|Int,Int,GtInt,Int,All,Int,Int,LtInt,All,Int,GtInt,Int,GtInt,GtInt,LtInt,Int,Int,Int,LtInt,Int,Int,Int,GtInt|GtInt,Int,LtInt,Int,LtInt,All,LtInt,LtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,GtInt,All,Int,GtInt,All|Int,Int,GtInt,All,LtInt,Int,Int,Int,GtInt,GtInt,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int|Int,Int,Int,GtInt,GtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,LtInt,LtInt,Int,All,LtInt|GtInt,Int,Int,LtInt,Int,Int,GtInt,LtInt,Int,Int,Int,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int|LtInt,All,Int,Int,Int,Int,Int,All,Int,LtInt,LtInt,All,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int|GtInt,Int,All,GtInt,LtInt,Int,Int,Int,All,GtInt,Int,Int,All,LtInt,Int,Int,Int,All,Int,Int,Int,All,Int|GtInt,Int,LtInt,GtInt,Int,Int,Int,Int,All,Int,All,Int,Int,GtInt,Int,Int,GtInt,LtInt,Int,All,LtInt,GtInt,Int|GtInt,Int,Int,LtInt,Int,Int,Int,Int,All,GtInt,GtInt,Int,All,GtInt,GtInt,Int,Int,Int,Int,Int,LtInt,Int,Int|Int,Int,Int,Int,Int,All,GtInt,Int,Int,Int,GtInt,Int,GtInt,Int,LtInt,Int,GtInt,Int,GtInt,Int,Int,Int,Int|Int,Int,LtInt,GtInt,LtInt,Int,Int,All,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,LtInt|LtInt,Int,LtInt,All,GtInt,Int,LtInt,Int,Int,GtInt,All,GtInt,Int,Int,LtInt,Int,Int,Int,All,Int,Int,LtInt,Int|Int,Int,LtInt,Int,All,Int,Int,Int,Int,Int,GtInt,Int,LtInt,Int,Int,Int,All,Int,Int,Int,GtInt,Int,Int|All,Int,Int,GtInt,Int,Int,Int,Int,LtInt,GtInt,Int,All,Int,LtInt,GtInt,GtInt,All,LtInt,GtInt,Int,Int,Int,Int|Int,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,All,All,Int|Int,Int,Int,Int,GtInt,Int,Int,LtInt,Int,All,All,GtInt,Int,Int,LtInt,LtInt,Int,LtInt,LtInt,LtInt,Int,GtInt,Int|Int,Int,GtInt,Int,All,Int,All,Int,Int,Int,Int,GtInt,Int,Int,GtInt,Int,LtInt,Int,Int,All,Int,Int,Int|LtInt,GtInt,Int,LtInt,Int,Int,Int,All,Int,GtInt,Int,Int,GtInt,GtInt,Int,Int,Int,Int,GtInt,LtInt,All,Int,GtInt|Int,Int,Int,All,Int,LtInt,Int,Int,GtInt,Int,GtInt,Int,LtInt,Int,Int,Int,Int,LtInt,Int,GtInt,LtInt,Int,Int|Int,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,GtInt,Int,GtInt,Int,GtInt,GtInt,Int,Int,Int,Int,Int,Int,GtInt|All,GtInt,LtInt,Int,Int,Int,LtInt,Int,All,Int,Int,Int,GtInt,Int,GtInt,Int,GtInt,All,Int,Int,Int,Int,GtInt|Int,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt,Int,GtInt,Int,GtInt,GtInt,Int,Int,GtInt,Int,Int,Int,Int,Int|All,LtInt,Int,Int,Int,LtInt,LtInt,Int,All,Int,GtInt,GtInt,All,Int,LtInt,Int,Int,GtInt,Int,Int,Int,GtInt,Int|Int,GtInt,Int,Int,Int,GtInt,Int,GtInt,All,Int,LtInt,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,GtInt,Int,Int|Int,Int,LtInt,Int,GtInt,Int,Int,Int,All,LtInt,Int,All,LtInt,LtInt,Int,Int,Int,Int,All,All,Int,Int,GtInt|Int,Int,Int,Int,All,Int,Int,Int,LtInt,LtInt,Int,Int,GtInt,Int,Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int|Int,GtInt,Int,Int,All,GtInt,GtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,Int,GtInt,Int,LtInt|LtInt,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,All,Int,Int,Int,Int,Int,LtInt,Int,Int,LtInt,Int,Int,Int|Int,Int,Int,LtInt,LtInt,GtInt,Int,Int,LtInt,LtInt,All,GtInt,Int,Int,All,Int,Int,Int,All,Int,Int,GtInt,LtInt|Int,Int,Int,Int,LtInt,LtInt,LtInt,Int,Int,GtInt,All,Int,Int,LtInt,Int,LtInt,Int,Int,All,Int,Int,All,All|All,Int,All,Int,Int,Int,All,All,Int,LtInt,Int,Int,Int,LtInt,GtInt,Int,GtInt,Int,Int,Int,Int,Int,LtInt|Int,Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,Int,Int,LtInt,Int,GtInt,Int,Int,Int|Int,LtInt,Int,All,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,All,LtInt,Int,Int,Int,GtInt,GtInt,Int,LtInt,GtInt|Int,LtInt,Int,GtInt,Int,Int,Int,LtInt,GtInt,LtInt,LtInt,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,Int,Int|Int,Int,Int,Int,Int,Int,GtInt,GtInt,GtInt,Int,Int,Int,GtInt,GtInt,Int,LtInt,Int,Int,Int,Int,All,LtInt,Int|LtInt,Int,All,GtInt,LtInt,All,Int,Int,Int,LtInt,Int,Int,Int,LtInt,Int,Int,GtInt,LtInt,Int,Int,Int,GtInt,GtInt|GtInt,LtInt,Int,GtInt,Int,Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt,LtInt,All,LtInt,GtInt,Int,LtInt|Int,Int,Int,LtInt,Int,LtInt,LtInt,Int,Int,Int,Int,GtInt,Int,Int,GtInt,Int,GtInt,Int,Int,Int,All,Int,Int|Int,Int,Int,GtInt,LtInt,GtInt,Int,GtInt,Int,LtInt,Int,LtInt,Int,Int,GtInt,LtInt,LtInt,GtInt,Int,Int,LtInt,LtInt,All|Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,GtInt,GtInt,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt|GtInt,Int,All,GtInt,Int,Int,Int,Int,All,Int,Int,Int,Int,All,GtInt,GtInt,All,GtInt,Int,Int,All,Int,GtInt|GtInt,GtInt,All,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,LtInt,All,Int,Int,Int,Int,LtInt,Int,All,GtInt,GtInt|]; 
constraint smart_table([x44,x35,x37,x19,x8,x38,x4,x36,x39,x42,x1,x43,x23,x31,x13,x25,x33,x10,x6,x7,x9,x5,x2],t0,signs0)::uniud;
constraint x44<84;
constraint not(x35=68);
constraint not(x37=75);
constraint not(x19=74);
constraint x8<148;
constraint x38>63;
constraint x4=127;
constraint not(x36=23);
constraint not(x39=173);
constraint not(x42=148);
constraint not(x1=13);
constraint not(x43=70);
constraint x23>110;
constraint x31<35;
constraint x13<98;
constraint not(x25=134);
constraint not(x33=123);
constraint x10>47;
constraint x6>101;
constraint not(x7=78);
constraint x9>78;
constraint not(x5=45);
constraint not(x2=188);
solve satisfy;