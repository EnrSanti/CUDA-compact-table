%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 72..175 : x0;
var 63..255 : x1;
var 64..99 : x2;
var 65..210 : x3;
var 60..257 : x4;
var 33..86 : x5;
var 42..243 : x6;
var 7..196 : x7;
var 18..32 : x8;
var 81..322 : x9;
var 65..174 : x10;
var 21..245 : x11;
var 55..291 : x12;
var 27..73 : x13;
var 11..125 : x14;
var 2..249 : x15;
var 48..225 : x16;
var 52..247 : x17;
var 22..259 : x18;
var 85..183 : x19;
var 54..174 : x20;
var 55..262 : x21;
var 21..215 : x22;
var 14..185 : x23;
var 66..290 : x24;
var 21..157 : x25;
var 42..286 : x26;
var 2..159 : x27;
var 73..265 : x28;
var 39..178 : x29;
var 3..141 : x30;
var 18..195 : x31;
var 4..82 : x32;
array [int,int] of 2..322 : t0=[|79,26,91,101,69,112,171,85,76,148,93,37,86,74|
50,67,210,40,58,239,164,261,188,163,86,123,44,175|
52,185,123,77,53,183,118,136,164,296,76,57,120,117|
106,17,146,117,50,161,169,141,217,303,79,23,13,64|
33,96,115,135,40,187,227,94,89,81,81,30,69,85|
45,108,187,131,32,197,148,61,135,164,95,127,95,218|
73,124,96,90,55,217,50,125,217,149,81,46,64,243|
135,107,74,121,37,202,107,259,168,186,91,61,34,177|
24,89,68,130,27,160,162,199,69,167,89,132,54,197|
34,90,80,44,43,60,56,142,156,140,75,14,66,153|
75,112,124,73,73,117,160,247,176,205,93,5,98,148|
148,75,136,242,62,186,249,134,239,137,79,78,111,85|
55,64,141,78,32,242,115,140,149,87,70,10,64,108|
32,55,166,241,48,192,97,264,94,124,97,138,109,194|
57,151,141,92,53,222,236,148,207,227,80,61,61,127|
87,176,174,63,50,174,264,167,241,202,68,124,107,196|
39,26,131,99,42,142,97,90,215,247,88,75,16,74|
56,176,65,54,44,83,186,222,116,90,97,85,13,232|
54,95,175,118,45,156,167,276,69,128,75,38,64,182|
55,83,125,215,29,199,205,108,198,241,73,39,91,198|
93,96,82,79,68,100,264,189,181,133,74,24,61,146|
137,13,98,167,41,197,141,183,135,303,74,28,121,101|
21,70,188,68,42,199,186,170,203,120,72,42,45,88|
101,61,107,184,54,81,246,283,188,92,90,64,71,73|
123,131,69,102,41,205,192,135,210,267,87,58,59,151|
143,51,70,185,73,169,217,185,238,300,86,25,47,135|
50,53,141,152,41,200,159,158,248,199,85,44,45,72|
66,15,115,202,64,250,123,206,213,263,85,64,76,103|
56,24,155,206,70,197,177,189,83,139,90,99,88,55|
75,188,100,215,63,248,44,175,252,139,64,74,42,136|
93,97,65,35,30,84,270,118,166,237,65,153,14,97|
95,59,106,180,47,198,67,220,206,177,97,23,38,138|
156,171,179,144,67,131,54,62,69,136,75,93,95,158|
96,39,134,198,64,101,185,191,84,101,73,91,72,146|
111,87,82,91,58,206,85,249,175,150,97,63,118,159|
54,168,87,168,36,196,197,167,225,119,81,58,83,82|
69,128,93,91,28,105,183,171,178,222,70,125,87,97|
47,76,70,31,56,113,111,163,215,96,85,5,41,145|
32,12,173,179,55,253,103,219,118,206,94,19,81,236|
70,136,207,163,39,126,108,236,255,193,82,31,62,115|
98,93,151,190,64,93,247,203,148,318,98,54,105,147|
26,156,96,157,60,207,49,126,112,322,70,155,30,107|
79,145,78,240,31,202,230,237,184,134,85,46,11,142|
118,73,168,170,32,219,141,251,141,151,91,157,57,177|
43,132,182,155,39,179,227,84,131,85,74,42,123,120|
125,65,104,202,30,190,122,176,101,287,94,28,31,179|
153,177,169,243,46,140,103,124,100,203,82,149,79,223|
40,170,126,95,34,115,235,265,179,130,91,114,68,58|
93,143,84,147,34,75,233,72,96,102,68,119,71,73|
111,89,193,239,57,218,110,64,205,109,80,4,108,149|
33,195,176,122,73,229,98,244,203,165,71,2,32,109|
84,119,88,233,33,88,72,195,154,119,74,151,24,227|
123,128,65,105,68,161,201,105,212,296,88,112,36,204|
107,157,167,140,32,75,240,133,97,320,92,67,80,119|
60,13,187,150,31,128,207,136,234,159,93,142,104,55|
120,84,123,78,35,185,267,228,194,246,80,119,54,215|
86,133,195,61,37,109,134,126,252,257,83,29,100,206|
23,178,69,105,33,187,176,159,246,238,99,101,30,52|
25,185,204,200,29,132,44,278,203,115,95,126,120,218|
25,145,185,66,50,156,145,105,181,218,90,28,113,76|
111,196,189,155,64,85,199,127,124,154,73,58,50,188|
32,71,178,100,46,158,75,153,90,285,96,4,87,102|
95,29,146,30,28,77,111,177,177,304,92,56,52,130|
98,140,89,146,60,69,172,132,187,299,80,142,37,212|
99,34,101,180,72,101,151,267,164,255,72,31,85,223|
42,78,176,194,45,60,197,187,159,276,89,142,81,164|
104,62,102,224,38,60,254,132,104,154,96,158,109,160|
114,79,80,174,57,83,241,200,113,142,86,98,59,178|
36,72,98,218,33,166,194,86,79,208,80,123,123,174|
98,48,92,157,30,197,183,252,199,95,86,36,108,67|
94,195,194,109,65,215,160,180,239,242,92,95,45,115|
29,196,147,141,53,96,77,211,225,286,73,4,25,239|
88,42,96,51,56,161,216,289,113,111,89,115,109,228|
69,67,192,135,61,87,272,134,247,255,69,141,75,114|
48,140,86,139,35,77,193,223,88,299,84,125,44,59|
63,112,169,229,52,239,142,179,130,148,79,111,103,215|
52,150,72,202,37,161,45,124,251,84,79,81,112,212|
99,62,87,91,58,177,43,271,166,194,68,124,125,207|
67,116,188,98,44,146,54,94,200,305,80,32,121,133|
99,82,99,114,51,168,236,268,162,256,77,9,35,74|
106,142,157,197,64,169,91,176,70,286,87,149,119,164|
119,90,98,83,37,114,280,163,182,125,93,73,73,245|
29,160,103,165,69,118,254,258,123,177,69,5,15,96|
120,105,174,240,46,162,185,163,214,275,89,16,27,233|
63,188,171,167,32,127,57,151,63,157,99,151,34,244|
137,137,68,148,64,203,204,229,171,252,92,9,110,125|
36,131,205,186,44,106,90,77,255,142,66,58,47,228|
85,129,157,62,27,237,142,192,250,161,92,53,90,180|
67,164,146,115,37,211,99,213,240,121,84,145,96,191|
67,92,162,106,45,90,150,199,182,243,96,134,118,246|
21,178,90,139,32,236,209,200,209,215,72,7,74,98|
49,116,159,93,67,175,70,89,130,123,93,36,25,82|
56,70,131,201,44,196,54,262,74,177,77,98,42,124|
151,55,182,168,72,69,272,285,185,123,65,95,88,76|
46,93,169,83,64,128,178,265,125,226,74,159,24,151|
143,160,192,113,50,138,92,287,133,204,76,101,67,135|
140,59,128,179,64,122,251,173,82,194,68,96,124,117|
113,121,83,51,34,62,274,240,176,276,68,137,92,137|
37,138,76,44,31,154,278,238,245,152,92,24,54,113|
23,192,158,124,44,83,126,104,225,166,97,91,38,59|
106,147,161,83,33,187,282,130,76,247,85,69,36,92|
38,180,184,129,27,178,136,288,73,152,82,121,38,235|
93,13,170,177,63,98,267,252,69,320,97,114,63,90|
31,32,150,241,60,241,211,279,75,113,95,111,78,79|
145,72,72,43,65,97,181,186,208,311,84,105,105,110|
146,189,166,170,62,92,195,133,185,183,98,18,14,170|
156,152,72,192,44,217,79,229,91,284,79,74,109,163|
133,188,177,157,62,137,64,220,97,287,85,50,65,82|
152,35,185,85,38,112,152,191,128,211,76,70,39,136|
48,186,174,201,30,77,88,276,104,277,85,64,35,131|
59,189,192,132,63,73,188,180,159,233,77,23,79,236|
96,142,114,149,28,239,154,265,224,138,64,22,73,73|
82,131,69,147,58,66,168,139,126,270,75,102,31,239|
127,67,100,144,58,167,122,271,211,166,69,108,107,187|
110,15,92,202,35,67,175,96,220,123,80,83,55,138|
28,190,107,185,40,250,59,100,170,130,64,74,19,160|
83,26,139,26,38,140,131,120,249,251,89,149,71,82|
113,96,205,149,49,153,260,179,87,271,70,120,62,62|
98,150,111,165,36,230,247,176,79,103,87,12,14,240|
120,106,197,49,59,228,189,212,200,123,81,43,67,166|
56,41,87,167,59,144,167,65,190,120,87,96,58,122|
87,136,106,175,33,79,247,114,249,152,82,29,65,186|
96,81,169,168,66,107,95,95,223,288,64,152,81,201|
38,69,176,149,39,236,83,141,89,258,81,103,51,114|
30,15,138,189,57,238,119,259,66,294,86,47,109,87|
136,63,192,195,48,87,74,261,254,198,68,139,22,54|
37,31,76,81,43,188,279,207,230,114,94,52,80,167|
138,184,144,108,28,85,81,219,182,227,98,70,54,190|
42,34,181,113,59,69,209,163,173,237,70,48,86,230|
60,104,187,144,54,103,168,221,221,217,74,132,108,139|
128,68,110,179,64,89,120,155,188,222,73,93,14,166|
68,93,134,88,55,144,167,143,157,95,93,84,92,83|
136,179,83,142,28,234,233,106,110,200,90,5,116,137|
140,166,95,221,39,185,139,172,159,274,83,63,92,106|
117,179,163,136,59,235,112,69,187,253,90,134,62,147|
40,155,187,210,58,234,78,232,241,92,74,154,123,58|
79,118,135,46,37,212,194,196,202,183,97,22,95,197|
114,55,93,217,49,137,243,211,86,285,89,8,67,59|
43,13,129,74,39,118,109,139,103,239,89,103,105,203|
124,64,204,242,27,195,175,66,230,228,99,11,76,160|
69,155,168,141,57,158,229,155,70,119,91,103,99,233|
98,122,133,190,49,230,93,56,112,187,75,19,98,162|
142,134,94,156,71,66,54,236,77,212,92,6,23,122|
23,125,185,184,62,153,262,244,150,106,76,77,121,170|
51,159,177,171,40,69,173,268,156,273,82,157,69,239|
104,182,136,58,37,116,171,250,94,197,95,35,46,85|
48,120,167,70,30,142,83,268,143,286,69,140,28,119|
63,193,141,106,49,206,74,86,198,230,74,157,95,138|
108,53,170,110,61,167,278,119,214,219,75,114,67,185|
110,81,105,226,60,129,119,100,203,243,98,80,97,155|
137,11,89,171,49,87,200,287,216,316,78,72,116,240|
109,154,94,68,69,216,69,162,115,298,88,50,107,144|
69,23,130,34,34,144,95,170,237,317,74,95,15,222|
32,66,93,191,61,141,210,203,193,148,72,51,37,143|
89,88,188,209,27,164,205,84,145,193,91,32,63,102|
49,122,164,233,67,256,206,215,239,195,97,96,104,92|
79,132,129,193,55,149,138,102,224,205,87,142,111,82|
84,60,160,119,42,85,236,59,83,100,81,34,11,53|
90,168,134,91,68,198,259,123,122,182,79,95,121,153|
95,115,136,176,37,223,122,174,93,303,76,29,86,228|
63,169,196,225,50,107,180,100,93,251,66,11,62,163|
144,91,144,189,59,61,276,217,65,313,91,15,36,172|
131,95,192,175,44,111,196,199,143,201,77,53,17,128|]; 
array [int,int] of SmartTableOp: signs0=[|Int,All,LtInt,LtInt,All,Int,Int,Int,Int,GtInt,Int,Int,Int,Int|Int,Int,LtInt,Int,Int,Int,All,Int,Int,Int,Int,Int,LtInt,Int|LtInt,LtInt,Int,Int,Int,Int,Int,Int,All,Int,Int,LtInt,GtInt,Int|Int,Int,Int,Int,Int,All,Int,Int,Int,Int,Int,GtInt,Int,Int|Int,GtInt,Int,Int,Int,Int,Int,Int,LtInt,GtInt,Int,LtInt,Int,GtInt|LtInt,GtInt,LtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,Int,Int|Int,LtInt,Int,LtInt,Int,GtInt,LtInt,Int,LtInt,GtInt,Int,Int,GtInt,All|Int,Int,LtInt,GtInt,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,All|LtInt,Int,Int,Int,GtInt,Int,Int,Int,GtInt,LtInt,Int,Int,GtInt,Int|Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,LtInt,LtInt,GtInt,Int|Int,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,LtInt,Int,Int,GtInt|LtInt,LtInt,Int,Int,LtInt,LtInt,LtInt,Int,LtInt,LtInt,LtInt,All,All,Int|LtInt,Int,Int,Int,GtInt,Int,GtInt,Int,Int,GtInt,Int,Int,LtInt,Int|Int,Int,LtInt,All,Int,Int,Int,GtInt,Int,Int,All,Int,Int,All|Int,LtInt,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,All,Int,GtInt|Int,LtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,All,Int,Int,Int|LtInt,Int,Int,LtInt,LtInt,Int,LtInt,Int,GtInt,GtInt,Int,Int,Int,Int|LtInt,Int,All,LtInt,Int,Int,LtInt,GtInt,Int,LtInt,GtInt,Int,Int,LtInt|Int,Int,Int,Int,LtInt,LtInt,Int,GtInt,Int,LtInt,Int,Int,Int,Int|Int,GtInt,Int,Int,Int,All,Int,Int,Int,Int,Int,Int,Int,Int|Int,Int,Int,LtInt,All,GtInt,All,GtInt,Int,Int,Int,LtInt,GtInt,Int|Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt|Int,Int,All,GtInt,Int,Int,All,Int,Int,Int,Int,Int,GtInt,Int|Int,All,Int,Int,Int,Int,LtInt,Int,All,Int,Int,Int,Int,Int|Int,LtInt,GtInt,All,LtInt,LtInt,LtInt,LtInt,Int,Int,LtInt,All,GtInt,GtInt|LtInt,Int,Int,Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int,Int,LtInt|GtInt,All,LtInt,All,Int,Int,Int,Int,GtInt,All,Int,Int,GtInt,Int|All,Int,GtInt,Int,Int,All,Int,LtInt,Int,Int,LtInt,LtInt,Int,Int|Int,Int,Int,GtInt,All,LtInt,GtInt,Int,Int,GtInt,Int,GtInt,Int,Int|GtInt,Int,Int,Int,Int,All,GtInt,Int,Int,Int,Int,Int,Int,GtInt|Int,Int,GtInt,Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,GtInt,GtInt|Int,Int,Int,Int,Int,Int,Int,All,GtInt,Int,Int,GtInt,LtInt,Int|Int,Int,Int,LtInt,All,LtInt,Int,Int,Int,Int,Int,Int,GtInt,All|Int,LtInt,All,Int,Int,Int,Int,Int,All,GtInt,GtInt,Int,Int,Int|Int,All,Int,GtInt,Int,Int,LtInt,Int,Int,Int,LtInt,GtInt,Int,GtInt|Int,Int,LtInt,Int,GtInt,Int,GtInt,Int,Int,Int,GtInt,All,LtInt,Int|Int,All,Int,Int,LtInt,Int,LtInt,Int,LtInt,All,Int,Int,Int,Int|Int,Int,Int,Int,LtInt,Int,Int,GtInt,LtInt,Int,LtInt,GtInt,Int,Int|LtInt,Int,Int,GtInt,LtInt,Int,GtInt,Int,Int,Int,GtInt,Int,Int,Int|GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,LtInt,GtInt,Int,Int|Int,GtInt,Int,All,GtInt,Int,LtInt,Int,LtInt,Int,Int,Int,All,Int|Int,LtInt,Int,Int,All,Int,Int,All,Int,GtInt,Int,Int,Int,LtInt|GtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,GtInt,LtInt,Int|GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int|GtInt,GtInt,LtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,All,Int|LtInt,Int,GtInt,Int,Int,Int,All,Int,All,Int,Int,GtInt,Int,Int|Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int|GtInt,GtInt,GtInt,LtInt,GtInt,Int,Int,LtInt,Int,Int,LtInt,LtInt,Int,Int|Int,GtInt,LtInt,Int,Int,GtInt,GtInt,Int,LtInt,Int,All,LtInt,Int,Int|Int,Int,LtInt,GtInt,LtInt,Int,Int,Int,Int,Int,All,Int,Int,All|LtInt,Int,GtInt,Int,Int,Int,LtInt,GtInt,Int,LtInt,GtInt,Int,Int,GtInt|LtInt,Int,Int,Int,Int,GtInt,Int,LtInt,Int,LtInt,LtInt,Int,Int,GtInt|Int,Int,Int,LtInt,All,Int,GtInt,All,Int,Int,Int,LtInt,Int,Int|Int,All,LtInt,GtInt,Int,Int,Int,LtInt,Int,Int,GtInt,Int,All,All|GtInt,LtInt,Int,GtInt,GtInt,Int,Int,Int,Int,Int,Int,LtInt,Int,Int|Int,Int,Int,Int,Int,LtInt,Int,GtInt,GtInt,LtInt,All,All,LtInt,Int|Int,Int,Int,Int,GtInt,LtInt,Int,Int,GtInt,LtInt,Int,GtInt,GtInt,Int|Int,All,Int,GtInt,GtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,Int|Int,All,Int,GtInt,All,Int,Int,Int,LtInt,Int,LtInt,Int,Int,All|Int,Int,Int,LtInt,LtInt,LtInt,LtInt,Int,GtInt,LtInt,LtInt,Int,All,LtInt|Int,All,Int,Int,All,Int,Int,LtInt,Int,GtInt,Int,Int,LtInt,Int|Int,Int,Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int,Int,GtInt,Int|Int,All,Int,Int,Int,GtInt,Int,Int,Int,Int,All,Int,GtInt,Int|Int,Int,Int,Int,Int,Int,Int,All,LtInt,Int,Int,All,GtInt,GtInt|GtInt,LtInt,GtInt,Int,Int,Int,All,Int,Int,All,GtInt,LtInt,Int,LtInt|Int,Int,Int,LtInt,Int,GtInt,Int,Int,Int,GtInt,Int,Int,Int,LtInt|Int,Int,GtInt,All,Int,GtInt,Int,Int,All,GtInt,Int,GtInt,Int,GtInt|GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int|Int,GtInt,GtInt,LtInt,Int,All,GtInt,All,Int,All,Int,All,Int,LtInt|Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,All,Int,Int|Int,Int,LtInt,LtInt,Int,Int,All,GtInt,Int,Int,All,Int,Int,GtInt|LtInt,Int,Int,GtInt,LtInt,Int,Int,LtInt,Int,LtInt,Int,All,Int,Int|Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int|Int,Int,GtInt,Int,Int,All,Int,All,Int,Int,All,LtInt,Int,GtInt|All,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,GtInt|GtInt,Int,Int,Int,LtInt,Int,Int,Int,Int,All,GtInt,Int,Int,GtInt|Int,Int,Int,Int,Int,Int,LtInt,GtInt,LtInt,Int,Int,All,Int,Int|Int,LtInt,Int,Int,GtInt,GtInt,Int,Int,Int,GtInt,Int,Int,LtInt,GtInt|GtInt,Int,LtInt,LtInt,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int|Int,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,LtInt,Int,GtInt,Int|GtInt,All,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int|Int,All,Int,GtInt,Int,Int,LtInt,Int,Int,Int,LtInt,Int,Int,Int|Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,GtInt|LtInt,LtInt,All,Int,Int,LtInt,GtInt,Int,GtInt,All,All,Int,Int,Int|All,Int,Int,GtInt,Int,Int,GtInt,All,Int,Int,Int,Int,Int,Int|Int,GtInt,All,LtInt,Int,GtInt,Int,Int,LtInt,LtInt,All,Int,All,Int|GtInt,Int,All,GtInt,Int,GtInt,LtInt,Int,Int,Int,LtInt,Int,Int,GtInt|Int,LtInt,GtInt,Int,Int,Int,LtInt,Int,GtInt,Int,Int,LtInt,Int,Int|Int,Int,GtInt,GtInt,Int,Int,Int,Int,All,Int,Int,LtInt,Int,Int|All,GtInt,GtInt,LtInt,Int,LtInt,Int,LtInt,Int,Int,Int,LtInt,Int,LtInt|Int,Int,Int,Int,GtInt,LtInt,Int,Int,LtInt,All,Int,Int,All,LtInt|Int,Int,Int,Int,GtInt,Int,Int,LtInt,LtInt,LtInt,Int,LtInt,Int,GtInt|Int,GtInt,LtInt,Int,Int,Int,All,Int,GtInt,Int,GtInt,GtInt,Int,Int|Int,Int,Int,Int,Int,Int,GtInt,All,LtInt,Int,Int,GtInt,GtInt,Int|Int,Int,Int,Int,All,Int,Int,GtInt,Int,Int,Int,Int,Int,All|Int,Int,LtInt,Int,Int,Int,Int,LtInt,LtInt,Int,All,All,GtInt,All|LtInt,Int,Int,LtInt,Int,LtInt,GtInt,Int,LtInt,Int,Int,Int,Int,Int|LtInt,Int,GtInt,GtInt,Int,Int,GtInt,GtInt,Int,Int,GtInt,Int,All,GtInt|All,Int,Int,Int,Int,Int,Int,Int,Int,All,LtInt,LtInt,Int,Int|Int,Int,Int,Int,Int,LtInt,All,Int,Int,Int,LtInt,LtInt,Int,Int|Int,Int,LtInt,Int,Int,Int,Int,Int,All,Int,Int,Int,LtInt,GtInt|Int,Int,Int,LtInt,LtInt,Int,GtInt,Int,Int,Int,Int,Int,LtInt,LtInt|Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,LtInt,GtInt,All,GtInt|Int,GtInt,LtInt,Int,LtInt,GtInt,GtInt,GtInt,LtInt,All,Int,Int,Int,Int|Int,Int,LtInt,GtInt,LtInt,Int,Int,GtInt,Int,All,Int,All,Int,Int|Int,Int,All,Int,Int,GtInt,All,Int,Int,Int,Int,GtInt,Int,Int|GtInt,GtInt,Int,Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int|LtInt,Int,GtInt,All,GtInt,All,Int,Int,LtInt,GtInt,Int,Int,Int,Int|Int,LtInt,Int,Int,Int,Int,All,GtInt,GtInt,Int,Int,Int,Int,LtInt|Int,Int,LtInt,LtInt,Int,Int,Int,LtInt,Int,Int,Int,All,Int,All|Int,All,Int,Int,Int,GtInt,Int,Int,Int,All,Int,Int,GtInt,LtInt|GtInt,All,Int,Int,All,LtInt,LtInt,LtInt,Int,Int,GtInt,GtInt,Int,Int|Int,Int,Int,LtInt,Int,Int,LtInt,Int,Int,GtInt,LtInt,All,Int,LtInt|All,Int,Int,LtInt,Int,GtInt,All,Int,LtInt,Int,Int,LtInt,Int,Int|LtInt,All,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int|Int,Int,LtInt,Int,All,GtInt,Int,Int,LtInt,GtInt,LtInt,GtInt,Int,GtInt|Int,All,Int,Int,LtInt,Int,GtInt,Int,All,Int,LtInt,LtInt,GtInt,Int|GtInt,Int,Int,GtInt,Int,All,Int,Int,Int,Int,Int,LtInt,Int,Int|Int,Int,All,GtInt,GtInt,Int,Int,Int,LtInt,Int,All,Int,Int,LtInt|Int,All,All,All,GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,Int|Int,GtInt,LtInt,Int,All,Int,Int,Int,LtInt,Int,GtInt,Int,Int,All|GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,LtInt,GtInt,Int,Int,GtInt|GtInt,Int,Int,LtInt,All,Int,GtInt,Int,Int,LtInt,All,All,Int,Int|LtInt,All,Int,Int,LtInt,LtInt,GtInt,Int,Int,LtInt,All,All,LtInt,Int|Int,Int,GtInt,Int,Int,GtInt,Int,Int,LtInt,LtInt,Int,LtInt,Int,All|Int,LtInt,Int,Int,Int,LtInt,Int,Int,Int,GtInt,All,Int,Int,LtInt|Int,GtInt,Int,GtInt,GtInt,GtInt,Int,All,Int,All,Int,Int,GtInt,Int|Int,Int,Int,Int,Int,Int,All,Int,Int,Int,All,Int,GtInt,All|GtInt,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int|Int,Int,LtInt,LtInt,All,GtInt,GtInt,All,Int,Int,GtInt,Int,LtInt,LtInt|Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int,Int,Int|GtInt,Int,Int,LtInt,LtInt,All,All,Int,GtInt,LtInt,GtInt,Int,LtInt,Int|Int,Int,Int,Int,Int,Int,Int,GtInt,GtInt,LtInt,Int,Int,LtInt,Int|All,GtInt,GtInt,Int,Int,Int,Int,Int,GtInt,All,Int,GtInt,LtInt,Int|LtInt,Int,LtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int|LtInt,LtInt,GtInt,Int,Int,GtInt,LtInt,GtInt,Int,GtInt,Int,Int,LtInt,LtInt|Int,GtInt,Int,All,LtInt,GtInt,Int,Int,LtInt,Int,Int,GtInt,GtInt,Int|Int,Int,All,All,Int,Int,Int,GtInt,Int,Int,All,Int,Int,LtInt|LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int|Int,Int,GtInt,GtInt,Int,Int,All,Int,Int,LtInt,Int,All,Int,LtInt|Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,LtInt,Int,GtInt|Int,Int,All,Int,Int,All,Int,GtInt,GtInt,Int,Int,GtInt,GtInt,All|Int,Int,Int,GtInt,Int,LtInt,All,GtInt,LtInt,LtInt,All,All,GtInt,Int|Int,LtInt,Int,All,All,Int,Int,Int,All,Int,LtInt,Int,LtInt,Int|Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,Int,Int,LtInt,LtInt,LtInt,GtInt|Int,Int,Int,LtInt,Int,All,Int,LtInt,All,Int,Int,All,Int,Int|Int,All,Int,GtInt,Int,Int,All,Int,Int,Int,Int,Int,Int,LtInt|LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,All,Int,Int,Int,Int|Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int|All,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,Int,LtInt,Int|Int,Int,Int,Int,Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int,Int|Int,GtInt,Int,Int,Int,Int,Int,Int,All,Int,All,LtInt,Int,Int|GtInt,Int,Int,All,Int,Int,GtInt,Int,All,Int,Int,GtInt,LtInt,Int|LtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,GtInt,GtInt|Int,GtInt,Int,GtInt,Int,LtInt,All,Int,All,Int,Int,Int,Int,LtInt|LtInt,Int,All,GtInt,LtInt,Int,Int,Int,All,GtInt,Int,GtInt,GtInt,Int|Int,Int,Int,Int,Int,Int,Int,GtInt,GtInt,Int,All,Int,Int,Int|Int,GtInt,Int,LtInt,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int|Int,Int,Int,LtInt,LtInt,All,GtInt,Int,GtInt,LtInt,LtInt,Int,Int,Int|Int,GtInt,Int,Int,Int,Int,Int,Int,LtInt,All,GtInt,Int,Int,GtInt|Int,Int,Int,GtInt,Int,All,GtInt,Int,Int,Int,Int,LtInt,Int,Int|GtInt,Int,Int,Int,LtInt,LtInt,Int,GtInt,Int,GtInt,Int,LtInt,All,Int|Int,All,Int,Int,Int,GtInt,Int,Int,LtInt,GtInt,Int,All,Int,Int|]; 
constraint smart_table([x25,x7,x3,x11,x13,x4,x26,x12,x1,x9,x2,x27,x14,x17],t0,signs0)::uniud;
constraint not(x25=126);
solve satisfy;