%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 71..122 : x0;
var 19..128 : x1;
var 29..76 : x2;
var 74..292 : x3;
var 53..69 : x4;
var 36..118 : x5;
var 11..203 : x6;
var 82..99 : x7;
var 5..21 : x8;
var 75..296 : x9;
var 62..299 : x10;
var 34..118 : x11;
var 19..72 : x12;
var 27..139 : x13;
var 80..232 : x14;
var 32..149 : x15;
var 84..120 : x16;
var 38..241 : x17;
var 36..157 : x18;
var 4..107 : x19;
var 74..206 : x20;
var 65..208 : x21;
var 58..180 : x22;
var 76..167 : x23;
var 39..201 : x24;
var 31..217 : x25;
var 25..144 : x26;
var 36..93 : x27;
var 31..270 : x28;
var 3..37 : x29;
var 79..158 : x30;
var 15..102 : x31;
var 14..126 : x32;
var 62..204 : x33;
var 42..134 : x34;
var 42..157 : x35;
var 21..193 : x36;
var 57..150 : x37;
var 39..127 : x38;
var 43..66 : x39;
var 85..126 : x40;
var 71..251 : x41;
var 28..167 : x42;
var 88..248 : x43;
var 63..184 : x44;
var 9..150 : x45;
var 74..287 : x46;
var 29..158 : x47;
array [int,int] of 3..299 : t0=[|188,143,79,106,147,94,151,126,175,94,14,81,70,105,91,250,128,35,84|
112,273,119,87,197,74,112,89,42,143,8,53,116,46,86,80,86,23,55|
45,220,54,106,109,64,101,242,151,79,21,42,64,156,98,223,61,3,97|
66,134,117,98,69,83,176,286,56,111,21,86,73,55,93,210,158,28,104|
120,200,63,113,147,76,64,198,50,129,15,72,31,132,91,91,68,26,49|
189,209,138,87,106,110,88,89,146,91,9,48,27,144,96,218,79,30,74|
115,163,111,106,112,113,67,129,156,111,19,63,147,66,88,71,126,27,67|
101,287,48,113,59,53,105,149,112,107,9,47,79,64,92,44,142,4,59|
121,285,97,118,172,103,153,83,187,92,12,86,150,121,90,255,146,23,116|
70,190,103,115,67,98,83,280,40,104,20,59,143,132,95,267,74,4,98|
188,145,134,89,95,84,177,280,68,83,9,88,24,116,84,77,85,30,95|
178,284,87,100,149,113,77,221,191,93,7,63,100,150,85,48,51,15,40|
90,259,60,118,66,94,170,285,94,81,10,74,16,49,84,164,158,32,61|
142,278,94,115,179,60,121,192,24,137,9,71,54,98,84,120,121,5,70|
106,76,92,119,165,45,75,175,96,68,18,90,9,139,82,183,148,22,42|
85,274,38,89,120,60,94,196,39,70,21,65,147,96,92,231,97,28,111|
193,281,69,117,49,63,168,166,59,66,15,89,111,101,88,189,62,5,68|
109,173,56,84,179,51,92,142,162,122,15,47,74,83,82,38,72,14,112|
85,189,109,100,164,86,115,207,185,105,17,40,43,106,85,157,122,17,63|
42,251,122,92,119,95,72,130,180,88,16,77,21,156,87,186,97,8,77|
88,136,132,120,131,49,117,132,192,76,15,85,114,94,96,94,95,25,82|
126,239,34,112,41,73,125,85,153,135,19,49,144,104,99,89,107,9,40|
164,221,31,86,56,113,178,278,53,148,8,73,61,74,89,66,64,26,82|
102,250,53,104,110,37,127,281,126,148,6,48,27,73,83,148,79,26,67|
114,190,58,87,66,43,125,174,38,74,15,53,108,32,82,193,100,32,94|
160,137,71,105,170,83,63,132,58,89,9,60,11,120,97,149,37,10,93|
100,135,121,116,67,76,154,248,30,119,9,85,130,105,84,122,63,23,115|
181,283,78,112,118,118,129,168,64,133,9,75,86,165,93,173,86,17,88|
40,124,88,115,150,110,81,116,151,76,11,73,108,47,93,242,72,27,67|
59,240,139,110,125,78,98,137,127,147,18,69,142,57,85,74,70,11,73|
182,189,41,112,68,118,158,213,22,128,12,86,82,146,85,82,137,7,51|
159,181,34,94,210,109,110,244,172,133,11,75,92,63,91,199,84,30,57|
150,286,79,87,196,95,164,108,160,89,9,36,118,158,99,188,69,3,100|
182,202,65,107,53,34,99,107,57,113,14,60,31,97,88,47,158,25,95|
118,215,47,101,154,84,116,253,85,101,12,62,54,142,94,170,32,27,80|
189,269,113,88,179,118,107,179,52,136,21,49,57,118,86,158,64,6,111|
102,114,88,90,179,114,129,114,72,57,17,44,149,73,93,148,87,31,82|
64,81,84,99,112,90,94,77,149,58,15,85,129,73,83,93,45,36,59|
142,219,60,95,210,34,173,178,114,146,5,75,84,153,88,189,110,6,97|
142,234,55,111,123,47,114,98,84,60,6,58,89,107,85,199,72,12,83|
101,209,121,100,86,116,94,239,103,125,10,79,115,54,82,245,43,25,114|
68,237,59,96,150,66,167,240,180,109,12,62,53,37,91,224,103,18,102|
108,213,80,110,44,42,134,118,128,60,11,51,23,73,86,112,108,28,82|
51,296,89,111,53,43,163,281,70,103,19,49,54,38,90,90,88,9,85|
129,231,111,99,216,92,82,286,39,75,13,88,115,124,96,31,73,16,41|
171,186,58,92,53,42,153,86,82,100,15,81,49,147,86,127,57,33,115|
110,140,32,87,113,95,66,135,180,80,21,67,52,101,99,89,62,17,104|
166,159,106,92,122,106,79,232,68,77,5,42,19,119,90,39,122,23,62|
78,187,91,94,129,84,150,278,146,102,13,92,95,72,95,184,31,24,118|
61,242,105,92,191,45,154,227,37,133,7,48,85,87,95,63,101,33,118|
190,112,126,92,56,110,142,236,63,118,17,85,120,105,84,237,103,35,57|
201,294,132,118,102,93,97,79,158,102,5,66,114,57,99,263,150,24,68|
135,114,54,117,173,57,173,130,108,100,5,71,126,101,85,89,142,37,87|
121,174,48,100,139,84,114,189,114,98,17,89,96,153,88,85,97,34,118|
194,267,46,91,211,48,118,76,122,138,14,49,71,43,92,103,42,36,84|
127,261,113,114,65,99,75,81,163,112,7,62,21,122,97,76,146,19,67|
47,136,43,102,46,109,115,161,100,67,14,93,53,47,94,60,32,34,71|
146,290,64,91,41,118,99,278,26,69,17,93,95,103,94,232,78,18,110|
153,152,103,111,153,72,128,185,145,139,9,81,100,43,83,172,147,23,55|
106,146,132,96,129,77,84,206,78,110,13,80,95,147,96,103,122,9,67|
126,155,38,89,171,71,137,151,75,78,12,66,31,94,99,97,50,9,110|
45,88,134,99,121,63,106,154,163,59,21,84,147,164,96,134,34,3,111|
92,107,75,95,109,34,167,278,59,99,7,71,86,67,87,198,133,13,65|
83,130,109,89,175,39,170,133,80,100,21,53,69,148,83,109,72,14,69|
74,293,60,86,168,67,118,252,181,137,12,69,97,140,91,212,96,30,52|
190,103,120,120,187,40,88,236,93,116,10,44,133,42,92,260,51,12,75|
116,188,102,120,181,35,174,221,153,86,6,67,119,79,87,137,106,8,82|
168,103,69,109,215,84,120,266,182,108,18,72,51,37,99,36,127,23,50|
53,127,58,110,157,102,106,260,123,115,6,41,24,63,89,55,141,6,82|
107,184,31,102,196,102,81,203,133,107,17,79,28,62,85,72,52,7,47|
173,251,114,89,73,112,118,169,41,104,21,67,99,162,91,168,147,35,57|
201,289,129,100,183,53,177,270,69,73,11,87,47,87,90,88,136,30,54|
109,236,93,90,197,99,133,238,72,59,21,54,118,71,97,163,76,33,36|
134,127,95,115,182,115,159,272,105,136,10,37,108,45,82,34,145,20,112|
90,169,92,92,148,59,70,165,126,133,21,48,9,51,90,71,62,32,79|
134,235,31,86,38,87,106,104,152,115,12,69,126,99,89,162,153,9,75|
115,108,87,98,70,116,92,81,52,72,19,78,16,72,82,105,55,6,80|
58,199,77,96,100,104,167,84,171,71,13,74,92,58,88,99,105,7,48|
110,262,84,85,160,111,77,224,112,144,10,44,81,61,89,122,58,13,95|
170,88,76,119,61,41,152,264,23,94,14,81,40,134,84,220,123,14,87|
62,88,118,118,182,38,158,84,66,79,16,64,88,29,88,259,78,21,67|
159,171,31,93,143,63,177,103,83,70,8,64,61,36,88,111,120,12,37|
59,81,97,89,85,105,66,122,116,91,18,73,108,77,97,170,42,28,106|
114,225,105,102,217,49,89,171,190,102,20,61,51,57,85,201,130,21,95|
100,144,71,103,82,53,139,108,21,72,21,80,62,69,99,94,55,16,93|
129,216,71,114,107,74,65,204,178,150,20,93,63,69,86,215,116,34,69|
51,196,79,100,165,48,134,178,93,150,11,50,112,126,93,177,107,11,118|
136,143,30,101,209,94,73,249,26,100,6,46,150,145,97,140,78,31,47|
169,266,32,101,93,96,114,96,136,136,13,62,97,49,93,125,87,30,99|
194,146,103,84,142,41,91,117,190,135,17,76,28,62,91,265,109,25,115|
192,196,105,114,126,103,78,91,165,65,13,54,75,66,89,250,105,12,53|
84,175,46,105,210,62,91,241,108,68,20,86,45,67,98,213,80,27,68|
115,290,56,85,161,77,95,120,53,59,12,78,94,142,84,106,30,17,43|]; 
array [int,int] of SmartTableOp: signs0=[|Int,Int,Int,GtInt,GtInt,Int,LtInt,LtInt,GtInt,GtInt,All,All,Int,LtInt,GtInt,GtInt,Int,Int,Int|Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,LtInt,Int,Int,GtInt|All,LtInt,Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,GtInt,Int,LtInt,Int|LtInt,Int,Int,Int,LtInt,Int,GtInt,Int,Int,LtInt,Int,Int,Int,All,Int,GtInt,Int,Int,Int|Int,LtInt,LtInt,Int,GtInt,Int,All,LtInt,Int,GtInt,LtInt,GtInt,Int,Int,GtInt,GtInt,Int,Int,Int|LtInt,All,Int,LtInt,Int,Int,Int,Int,LtInt,LtInt,All,Int,Int,Int,Int,Int,Int,GtInt,Int|LtInt,All,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,All,Int,Int,Int,Int,LtInt|All,Int,Int,Int,Int,Int,Int,All,Int,Int,Int,Int,LtInt,Int,LtInt,Int,Int,Int,LtInt|Int,Int,Int,GtInt,GtInt,All,GtInt,Int,Int,Int,LtInt,All,Int,Int,LtInt,GtInt,Int,GtInt,Int|Int,Int,All,Int,Int,Int,All,All,Int,Int,LtInt,GtInt,Int,GtInt,Int,Int,Int,Int,Int|Int,Int,Int,Int,LtInt,LtInt,Int,Int,Int,GtInt,Int,LtInt,Int,Int,LtInt,Int,Int,Int,Int|Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,All,GtInt|Int,GtInt,GtInt,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt,Int,Int,Int,Int,LtInt|Int,All,All,Int,Int,Int,LtInt,LtInt,GtInt,Int,All,All,Int,All,LtInt,Int,All,All,LtInt|Int,Int,Int,All,Int,Int,Int,LtInt,GtInt,GtInt,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int|Int,Int,LtInt,LtInt,GtInt,Int,Int,Int,LtInt,All,Int,Int,All,Int,LtInt,Int,Int,All,All|All,Int,LtInt,Int,Int,Int,LtInt,All,Int,Int,Int,GtInt,Int,Int,GtInt,Int,Int,Int,All|Int,Int,Int,All,GtInt,GtInt,LtInt,All,Int,Int,Int,Int,Int,All,Int,GtInt,LtInt,Int,All|GtInt,GtInt,LtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,GtInt|Int,Int,GtInt,Int,Int,All,Int,Int,Int,LtInt,Int,Int,All,Int,Int,GtInt,LtInt,LtInt,All|Int,Int,LtInt,GtInt,Int,GtInt,GtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,GtInt|Int,GtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,All,GtInt,Int,Int,Int,Int,Int,LtInt|Int,Int,LtInt,Int,All,All,GtInt,LtInt,Int,Int,GtInt,Int,Int,Int,All,Int,GtInt,GtInt,Int|Int,Int,Int,All,GtInt,GtInt,GtInt,Int,LtInt,GtInt,LtInt,Int,GtInt,Int,Int,Int,LtInt,Int,Int|Int,GtInt,Int,Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,All,LtInt,LtInt,Int|Int,Int,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,Int,GtInt,LtInt|Int,All,Int,Int,GtInt,GtInt,Int,All,Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,LtInt,Int,Int|All,All,GtInt,All,Int,LtInt,GtInt,All,Int,Int,Int,Int,LtInt,Int,Int,LtInt,Int,Int,GtInt|Int,LtInt,All,LtInt,Int,LtInt,Int,GtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,GtInt,LtInt,Int,LtInt,GtInt,Int,LtInt,All,Int,Int,Int,GtInt,Int,Int,All,Int,GtInt,All|LtInt,LtInt,Int,GtInt,All,Int,Int,LtInt,LtInt,GtInt,LtInt,Int,Int,GtInt,All,Int,Int,Int,Int|Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,LtInt,LtInt,LtInt,Int,Int,Int,Int,Int,All,Int,Int|Int,Int,Int,GtInt,Int,Int,Int,GtInt,Int,Int,GtInt,Int,GtInt,Int,LtInt,LtInt,All,Int,Int|Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,All,All,All,Int,Int,Int,Int,Int,Int|Int,Int,Int,Int,Int,Int,GtInt,All,Int,LtInt,GtInt,Int,All,Int,GtInt,Int,All,Int,Int|Int,Int,LtInt,Int,Int,Int,Int,Int,Int,All,Int,Int,GtInt,Int,GtInt,LtInt,Int,Int,Int|Int,Int,Int,Int,Int,Int,LtInt,Int,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,LtInt,Int,Int,Int,Int,All,Int,LtInt,LtInt,GtInt,Int,Int,Int,LtInt,Int,GtInt,GtInt,LtInt|Int,LtInt,Int,Int,Int,Int,LtInt,LtInt,Int,GtInt,LtInt,Int,All,Int,GtInt,Int,LtInt,Int,Int|Int,Int,All,Int,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,LtInt|Int,Int,Int,All,All,Int,All,Int,LtInt,Int,GtInt,GtInt,Int,Int,Int,LtInt,Int,Int,Int|GtInt,Int,Int,Int,Int,All,Int,Int,LtInt,All,Int,All,Int,Int,Int,Int,Int,GtInt,All|GtInt,GtInt,GtInt,All,LtInt,Int,Int,Int,Int,All,LtInt,Int,LtInt,Int,Int,Int,Int,Int,Int|All,GtInt,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,GtInt,Int,Int,Int,GtInt,LtInt,Int,All|LtInt,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,GtInt,Int,All,Int,GtInt,Int,Int|Int,LtInt,LtInt,Int,GtInt,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int,LtInt,LtInt,GtInt,Int|Int,GtInt,Int,All,LtInt,Int,Int,GtInt,GtInt,Int,Int,LtInt,LtInt,LtInt,GtInt,Int,Int,Int,Int|Int,Int,Int,LtInt,Int,Int,LtInt,Int,GtInt,All,All,Int,Int,Int,Int,Int,Int,GtInt,All|Int,Int,Int,Int,Int,Int,GtInt,Int,Int,GtInt,All,Int,LtInt,LtInt,All,Int,Int,LtInt,GtInt|Int,GtInt,Int,Int,LtInt,GtInt,LtInt,Int,LtInt,Int,All,Int,Int,GtInt,GtInt,Int,LtInt,Int,Int|Int,Int,LtInt,GtInt,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,All|Int,LtInt,Int,Int,Int,LtInt,Int,LtInt,Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,GtInt,LtInt,Int|Int,Int,Int,Int,Int,LtInt,GtInt,All,Int,Int,LtInt,Int,Int,Int,Int,All,Int,Int,LtInt|All,Int,Int,Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,All,All,Int,GtInt,Int|All,Int,Int,Int,Int,LtInt,Int,Int,LtInt,GtInt,Int,Int,Int,LtInt,Int,LtInt,GtInt,All,GtInt|All,LtInt,GtInt,Int,LtInt,Int,Int,Int,Int,Int,All,Int,All,GtInt,GtInt,Int,Int,Int,Int|GtInt,Int,LtInt,All,Int,Int,Int,Int,GtInt,Int,LtInt,GtInt,Int,Int,LtInt,LtInt,Int,Int,Int|All,Int,Int,Int,Int,GtInt,All,All,Int,Int,Int,GtInt,GtInt,GtInt,Int,Int,GtInt,Int,GtInt|GtInt,GtInt,GtInt,GtInt,Int,Int,Int,GtInt,Int,All,LtInt,Int,Int,Int,Int,Int,All,Int,GtInt|Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,All,Int,GtInt,Int,Int,Int,GtInt,LtInt|GtInt,GtInt,Int,GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,All,Int,Int,Int,GtInt,Int|Int,All,GtInt,Int,GtInt,GtInt,LtInt,Int,All,Int,LtInt,LtInt,All,GtInt,GtInt,Int,Int,Int,Int|GtInt,Int,GtInt,Int,Int,LtInt,Int,Int,LtInt,All,All,Int,Int,LtInt,GtInt,LtInt,Int,All,LtInt|LtInt,Int,Int,Int,Int,All,Int,All,Int,LtInt,Int,Int,GtInt,Int,Int,Int,All,GtInt,Int|Int,LtInt,LtInt,Int,LtInt,Int,GtInt,LtInt,Int,All,All,Int,Int,Int,Int,GtInt,All,Int,Int|LtInt,Int,Int,LtInt,All,Int,Int,Int,LtInt,GtInt,Int,LtInt,Int,LtInt,Int,Int,All,Int,GtInt|Int,Int,Int,Int,Int,GtInt,LtInt,Int,GtInt,LtInt,LtInt,LtInt,Int,LtInt,Int,Int,LtInt,Int,Int|Int,All,All,LtInt,GtInt,All,Int,Int,GtInt,Int,GtInt,Int,Int,LtInt,Int,GtInt,All,All,LtInt|All,GtInt,Int,Int,Int,Int,LtInt,All,GtInt,Int,LtInt,LtInt,Int,Int,Int,LtInt,Int,Int,Int|Int,Int,LtInt,Int,LtInt,GtInt,Int,Int,All,Int,All,GtInt,Int,GtInt,Int,Int,Int,Int,GtInt|Int,All,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,LtInt,GtInt,Int,All,Int,LtInt,GtInt,Int|Int,All,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,LtInt,GtInt,Int|Int,Int,Int,Int,Int,LtInt,LtInt,GtInt,All,GtInt,LtInt,Int,GtInt,LtInt,GtInt,LtInt,All,Int,Int|GtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int,Int,LtInt,GtInt,GtInt,All,LtInt,Int,Int,Int,LtInt|GtInt,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,GtInt,LtInt,Int,GtInt,Int,GtInt,GtInt,Int,Int|Int,LtInt,Int,All,All,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,LtInt,All,Int,GtInt,Int|Int,LtInt,LtInt,Int,LtInt,Int,Int,LtInt,Int,Int,LtInt,GtInt,All,Int,Int,GtInt,Int,GtInt,GtInt|Int,Int,Int,All,Int,Int,Int,LtInt,LtInt,GtInt,All,GtInt,Int,GtInt,Int,GtInt,Int,GtInt,Int|LtInt,Int,Int,All,LtInt,Int,Int,All,Int,LtInt,All,Int,Int,GtInt,GtInt,Int,GtInt,LtInt,Int|Int,LtInt,Int,Int,Int,GtInt,LtInt,Int,GtInt,Int,LtInt,GtInt,GtInt,Int,GtInt,Int,GtInt,Int,LtInt|LtInt,Int,Int,Int,LtInt,GtInt,GtInt,LtInt,GtInt,LtInt,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int|Int,Int,Int,Int,LtInt,LtInt,GtInt,Int,Int,GtInt,All,LtInt,Int,LtInt,All,Int,Int,Int,GtInt|GtInt,GtInt,Int,Int,LtInt,GtInt,All,Int,LtInt,Int,Int,GtInt,GtInt,Int,LtInt,GtInt,Int,GtInt,GtInt|Int,Int,Int,Int,GtInt,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,Int,All,All,Int,Int|GtInt,GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt,All,Int,GtInt,LtInt|Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,Int,All,Int,Int,Int,Int|Int,Int,GtInt,GtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,All,Int|GtInt,LtInt,GtInt,Int,LtInt,All,Int,Int,LtInt,Int,Int,GtInt,LtInt,Int,Int,All,Int,Int,LtInt|GtInt,Int,All,Int,Int,LtInt,All,Int,GtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,Int,LtInt,Int,All,GtInt,Int,GtInt,Int,GtInt,Int,Int,Int,Int,LtInt,Int,LtInt,Int,GtInt|Int,GtInt,Int,Int,GtInt,Int,Int,Int,GtInt,Int,Int,All,Int,All,Int,Int,All,GtInt,All|Int,GtInt,Int,Int,Int,Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int|]; 
constraint smart_table([x24,x9,x13,x16,x25,x11,x44,x46,x36,x37,x8,x27,x45,x42,x7,x28,x47,x29,x5],t0,signs0)::gpu;
constraint not(x24=121);
constraint not(x9=292);
constraint not(x13=27);
solve satisfy;