%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 82..298 : x0;
var 18..168 : x1;
var 56..153 : x2;
var 37..115 : x3;
var 63..173 : x4;
var 55..173 : x5;
var 2..134 : x6;
var 35..205 : x7;
var 9..106 : x8;
var 83..311 : x9;
var 18..132 : x10;
var 19..106 : x11;
var 9..199 : x12;
var 56..88 : x13;
var 1..133 : x14;
var 9..42 : x15;
var 7..171 : x16;
var 27..67 : x17;
var 45..249 : x18;
var 59..232 : x19;
var 60..242 : x20;
var 70..82 : x21;
var 61..308 : x22;
var 62..255 : x23;
var 62..156 : x24;
var 42..261 : x25;
var 75..87 : x26;
var 64..224 : x27;
var 24..210 : x28;
var 40..85 : x29;
var 77..232 : x30;
var 12..243 : x31;
var 77..194 : x32;
var 39..56 : x33;
var 18..68 : x34;
var 54..183 : x35;
var 7..251 : x36;
var 26..153 : x37;
var 4..228 : x38;
var 77..172 : x39;
var 79..218 : x40;
var 73..183 : x41;
var 3..87 : x42;
array [int,int] of 1..311 : t0=[|114,192,90,42,58,28,58,116,157,149,94,145,128,236,138,225|
155,271,75,61,103,14,78,97,106,130,150,78,118,119,164,70|
229,223,120,50,89,21,61,56,89,138,69,79,232,216,90,222|
37,227,180,45,44,35,52,113,149,148,115,85,222,189,67,221|
130,169,189,38,107,28,41,114,170,39,145,78,61,192,123,118|
124,269,184,45,104,64,74,99,139,136,73,98,163,78,72,158|
173,81,219,31,104,71,62,22,141,98,115,119,225,72,75,182|
186,79,115,51,79,72,71,28,98,89,145,79,177,133,118,74|
12,81,193,47,38,22,57,82,61,185,152,128,126,160,69,140|
154,69,112,32,53,15,43,19,126,163,119,151,123,201,87,93|
152,78,148,33,114,56,65,118,139,15,151,86,217,65,100,208|
120,164,69,51,56,40,69,108,125,68,114,135,235,193,131,199|
225,273,92,27,94,82,50,62,142,142,87,123,70,196,82,230|
202,249,79,49,43,68,68,117,149,32,67,124,109,124,63,151|
152,260,119,63,71,18,54,123,138,81,95,62,117,155,162,101|
104,204,171,60,62,4,80,90,156,43,111,93,72,83,63,68|
113,121,111,48,49,21,48,95,113,189,68,130,119,152,147,202|
31,98,204,44,92,49,63,24,95,29,141,105,118,223,90,100|
239,184,214,51,113,16,43,58,107,179,95,127,235,225,128,69|
105,149,90,63,50,22,56,121,71,22,86,92,227,116,155,232|
68,257,145,45,66,27,54,115,140,21,72,118,139,72,120,95|
160,102,183,52,84,69,47,45,81,75,152,98,86,152,75,61|
97,61,136,54,101,54,72,104,172,127,79,135,96,188,130,95|
204,118,144,28,48,57,58,43,155,34,131,148,91,178,131,116|
235,96,65,54,38,81,80,67,124,142,94,146,226,62,129,76|
69,252,115,56,93,11,84,107,127,76,81,129,129,70,168,222|
45,252,97,49,50,81,73,105,99,174,108,64,213,189,84,192|
24,262,92,64,115,52,47,66,61,99,68,72,213,203,168,220|
141,216,131,30,69,79,83,76,86,15,133,143,119,170,95,158|
102,269,157,48,111,78,81,103,66,27,150,148,131,226,170,131|
150,296,86,34,105,45,54,102,73,196,124,90,238,91,156,187|
25,74,104,27,83,69,81,88,103,17,123,109,211,233,86,205|
106,97,197,53,112,24,48,122,129,153,60,74,163,246,76,152|
93,89,139,30,110,70,82,36,111,173,63,62,159,181,113,94|
58,226,160,54,96,24,74,72,113,99,110,148,195,85,117,194|
143,178,147,64,89,3,45,117,60,92,120,104,207,230,104,89|
71,111,120,34,75,71,85,128,59,16,141,72,187,65,74,231|
167,227,126,31,60,56,84,51,138,51,97,119,186,211,141,74|
187,62,138,37,82,25,45,57,85,39,94,126,154,96,102,156|
41,106,200,47,70,5,48,24,92,45,63,154,224,82,173,100|
213,271,122,55,66,30,77,63,121,91,111,82,90,103,156,77|
195,112,130,35,84,87,69,74,78,148,74,133,213,127,77,92|
29,68,132,52,113,46,54,30,135,45,104,152,79,130,98,203|
60,265,79,44,89,48,76,79,148,144,127,115,143,246,163,78|
103,107,184,31,37,59,44,77,65,93,63,97,231,64,132,102|
88,123,216,41,37,52,82,49,155,160,128,70,149,91,74,115|
99,250,216,30,55,79,63,99,141,22,146,106,242,203,111,83|
47,296,178,27,64,50,49,99,83,173,141,85,60,223,69,66|
210,105,108,36,112,11,59,61,123,182,56,66,149,127,70,106|
168,263,119,45,71,46,56,110,170,27,101,63,199,254,164,160|
180,179,114,43,92,78,48,49,65,48,75,123,77,159,97,152|
123,256,101,34,55,66,72,32,169,136,73,116,164,115,120,171|
150,251,145,61,55,43,58,35,171,174,127,126,66,183,131,62|
78,183,193,34,101,20,67,115,62,57,119,62,242,122,100,214|
68,138,139,42,98,42,65,101,102,199,98,149,166,134,82,184|
170,72,139,40,62,76,48,96,133,104,114,106,99,102,152,112|
61,277,75,61,74,43,40,113,108,199,140,154,69,252,109,193|
220,306,193,67,100,57,71,99,118,165,126,138,132,202,154,95|
100,76,189,35,97,75,62,57,75,102,71,79,136,130,98,101|
203,189,189,35,73,32,63,60,120,52,118,64,82,114,112,61|
165,166,119,31,37,8,80,29,138,138,135,150,209,133,130,192|
77,211,83,47,107,78,70,115,109,48,116,117,75,99,125,145|
207,87,159,56,109,7,71,28,155,13,152,83,62,248,128,209|
26,213,90,45,88,16,76,122,155,31,81,103,139,76,83,126|
59,87,146,43,72,53,40,124,160,114,112,72,108,207,102,89|
152,70,162,60,49,66,47,79,101,78,68,67,195,195,145,169|
33,292,101,58,99,87,80,125,83,85,71,80,135,246,75,137|
12,189,192,60,80,85,85,125,114,125,80,100,149,241,71,124|
48,289,203,30,91,52,43,57,90,112,146,99,61,101,128,115|
59,117,141,35,107,54,73,125,76,44,111,74,109,238,141,94|
116,283,171,63,101,77,57,82,156,44,69,101,129,219,76,199|
188,186,109,58,66,72,45,98,140,184,103,144,213,103,84,164|
24,161,152,58,43,54,73,98,142,136,139,112,128,227,133,129|
14,218,216,66,49,70,72,82,156,148,74,141,72,179,154,78|
95,93,119,57,74,4,59,93,109,43,136,74,105,197,172,203|]; 
array [int,int] of SmartTableOp: signs0=[|GtInt,Int,Int,LtInt,All,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,GtInt,GtInt|Int,GtInt,Int,All,Int,Int,GtInt,Int,Int,LtInt,LtInt,LtInt,GtInt,Int,GtInt,GtInt|LtInt,LtInt,All,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int|LtInt,Int,All,LtInt,Int,Int,Int,GtInt,All,Int,Int,GtInt,Int,LtInt,LtInt,All|LtInt,Int,Int,LtInt,Int,Int,GtInt,LtInt,Int,LtInt,All,Int,Int,GtInt,Int,Int|LtInt,Int,GtInt,Int,Int,Int,Int,GtInt,All,LtInt,Int,Int,Int,LtInt,Int,Int|LtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,All,GtInt,Int,GtInt,Int,Int,Int|GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int|All,Int,GtInt,Int,LtInt,GtInt,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,LtInt|Int,GtInt,Int,Int,LtInt,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,Int|Int,Int,Int,Int,All,All,Int,Int,LtInt,All,LtInt,Int,LtInt,All,Int,GtInt|All,Int,Int,Int,Int,Int,GtInt,Int,Int,LtInt,Int,GtInt,Int,Int,LtInt,GtInt|All,All,GtInt,Int,LtInt,Int,LtInt,Int,All,GtInt,Int,Int,All,Int,Int,Int|LtInt,LtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int,Int,Int|Int,GtInt,Int,LtInt,Int,All,GtInt,Int,All,Int,Int,All,All,Int,Int,GtInt|Int,LtInt,Int,Int,All,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,All|LtInt,Int,Int,GtInt,LtInt,GtInt,Int,Int,LtInt,GtInt,Int,Int,All,LtInt,Int,Int|LtInt,Int,Int,Int,Int,All,Int,Int,GtInt,LtInt,GtInt,Int,GtInt,Int,Int,Int|GtInt,Int,Int,Int,GtInt,Int,GtInt,All,Int,Int,LtInt,Int,Int,Int,Int,Int|Int,Int,All,Int,LtInt,LtInt,All,Int,Int,Int,Int,Int,Int,Int,GtInt,Int|Int,Int,Int,Int,Int,All,Int,All,Int,GtInt,All,Int,Int,Int,LtInt,Int|GtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int,GtInt,LtInt,All,Int,LtInt,Int,GtInt|Int,GtInt,Int,LtInt,GtInt,LtInt,GtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,All|Int,Int,All,Int,Int,Int,LtInt,Int,LtInt,Int,Int,Int,All,Int,Int,Int|LtInt,Int,LtInt,All,All,Int,All,Int,GtInt,LtInt,All,GtInt,Int,LtInt,GtInt,LtInt|All,All,LtInt,Int,Int,Int,Int,GtInt,Int,All,Int,Int,LtInt,All,Int,Int|Int,LtInt,GtInt,Int,Int,LtInt,LtInt,LtInt,LtInt,LtInt,Int,GtInt,Int,Int,Int,LtInt|Int,All,LtInt,LtInt,GtInt,Int,LtInt,Int,LtInt,LtInt,Int,Int,Int,LtInt,LtInt,GtInt|Int,GtInt,GtInt,Int,LtInt,All,LtInt,LtInt,All,Int,LtInt,All,Int,Int,GtInt,Int|Int,All,Int,Int,All,GtInt,Int,Int,All,Int,LtInt,Int,Int,All,Int,Int|LtInt,Int,LtInt,Int,Int,GtInt,Int,LtInt,Int,All,LtInt,GtInt,Int,Int,All,Int|Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,LtInt|Int,GtInt,Int,LtInt,Int,LtInt,Int,Int,Int,Int,All,LtInt,Int,GtInt,Int,Int|Int,GtInt,Int,Int,All,LtInt,Int,Int,LtInt,LtInt,Int,GtInt,GtInt,Int,Int,Int|LtInt,Int,Int,GtInt,Int,Int,LtInt,Int,GtInt,Int,Int,LtInt,LtInt,LtInt,GtInt,Int|Int,Int,All,Int,Int,Int,Int,GtInt,All,Int,LtInt,Int,GtInt,GtInt,All,GtInt|Int,GtInt,GtInt,LtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int,LtInt,All,Int,Int|GtInt,Int,LtInt,Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt,Int|Int,GtInt,LtInt,All,Int,Int,Int,LtInt,GtInt,LtInt,Int,Int,Int,Int,All,GtInt|GtInt,GtInt,GtInt,Int,GtInt,Int,GtInt,LtInt,GtInt,Int,Int,GtInt,Int,All,Int,Int|LtInt,Int,Int,Int,LtInt,Int,LtInt,GtInt,GtInt,GtInt,Int,Int,LtInt,LtInt,Int,Int|GtInt,Int,GtInt,Int,Int,Int,Int,GtInt,Int,GtInt,All,Int,Int,Int,Int,Int|Int,LtInt,GtInt,Int,Int,Int,All,Int,All,Int,Int,Int,Int,LtInt,Int,GtInt|All,Int,Int,Int,Int,Int,All,Int,Int,Int,Int,LtInt,Int,Int,Int,Int|LtInt,Int,Int,Int,Int,LtInt,LtInt,Int,Int,Int,All,Int,Int,LtInt,LtInt,GtInt|Int,GtInt,LtInt,Int,Int,All,All,Int,Int,LtInt,Int,Int,Int,All,Int,GtInt|Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,Int,GtInt,All,Int,All|GtInt,All,Int,Int,LtInt,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int|Int,Int,LtInt,GtInt,Int,Int,GtInt,GtInt,All,GtInt,LtInt,GtInt,All,Int,All,Int|Int,All,GtInt,LtInt,LtInt,Int,LtInt,GtInt,All,Int,GtInt,LtInt,Int,All,Int,GtInt|LtInt,GtInt,Int,Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int|Int,LtInt,Int,LtInt,LtInt,GtInt,GtInt,Int,All,GtInt,Int,GtInt,Int,Int,Int,Int|LtInt,All,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,All,GtInt|Int,Int,LtInt,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,LtInt,GtInt|GtInt,Int,Int,Int,Int,Int,LtInt,Int,GtInt,Int,GtInt,Int,Int,LtInt,Int,Int|LtInt,Int,LtInt,Int,LtInt,Int,Int,Int,GtInt,Int,Int,All,Int,Int,Int,Int|Int,Int,GtInt,Int,Int,Int,Int,Int,Int,All,Int,Int,Int,GtInt,Int,GtInt|Int,Int,LtInt,Int,Int,Int,Int,GtInt,LtInt,All,Int,Int,GtInt,Int,Int,LtInt|Int,Int,Int,Int,GtInt,LtInt,Int,LtInt,LtInt,All,Int,Int,Int,LtInt,Int,Int|LtInt,Int,Int,GtInt,GtInt,Int,Int,LtInt,GtInt,LtInt,All,Int,All,Int,GtInt,GtInt|Int,Int,LtInt,LtInt,Int,All,Int,LtInt,Int,Int,Int,LtInt,LtInt,All,GtInt,Int|Int,Int,LtInt,Int,Int,Int,GtInt,LtInt,Int,Int,GtInt,Int,LtInt,GtInt,GtInt,GtInt|Int,All,GtInt,All,GtInt,Int,LtInt,Int,LtInt,Int,Int,Int,GtInt,Int,LtInt,Int|Int,GtInt,Int,Int,Int,GtInt,Int,All,All,Int,All,Int,Int,Int,Int,GtInt|GtInt,Int,Int,GtInt,Int,Int,LtInt,Int,All,Int,LtInt,LtInt,All,Int,Int,Int|Int,Int,Int,LtInt,LtInt,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,GtInt|Int,Int,Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,LtInt,GtInt,Int,GtInt,Int|LtInt,Int,Int,All,Int,Int,Int,GtInt,Int,LtInt,LtInt,All,LtInt,Int,LtInt,Int|Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,All,LtInt,LtInt,Int,GtInt,Int|All,Int,LtInt,Int,LtInt,GtInt,GtInt,Int,Int,GtInt,GtInt,GtInt,LtInt,Int,GtInt,LtInt|GtInt,GtInt,GtInt,Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,GtInt|Int,LtInt,All,LtInt,GtInt,Int,Int,Int,Int,Int,Int,All,Int,LtInt,LtInt,Int|Int,Int,Int,Int,All,Int,GtInt,GtInt,Int,LtInt,Int,LtInt,Int,Int,Int,LtInt|GtInt,Int,Int,LtInt,Int,Int,Int,All,LtInt,Int,Int,GtInt,Int,Int,Int,GtInt|Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,LtInt,LtInt,LtInt,All|]; 
constraint smart_table([x31,x22,x27,x17,x3,x42,x29,x10,x5,x12,x2,x24,x20,x23,x4,x19],t0,signs0)::uniud;
solve satisfy;