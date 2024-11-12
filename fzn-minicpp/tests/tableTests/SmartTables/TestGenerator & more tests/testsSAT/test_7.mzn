%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 55..184 : x0;
var 16..138 : x1;
var 3..181 : x2;
var 79..257 : x3;
var 44..249 : x4;
var 29..241 : x5;
var 58..124 : x6;
var 14..64 : x7;
var 38..164 : x8;
var 41..209 : x9;
var 47..230 : x10;
var 77..140 : x11;
var 33..44 : x12;
var 50..63 : x13;
var 23..86 : x14;
var 87..255 : x15;
var 67..109 : x16;
var 57..139 : x17;
var 40..161 : x18;
var 49..108 : x19;
var 47..261 : x20;
var 14..171 : x21;
var 73..145 : x22;
var 54..145 : x23;
var 84..281 : x24;
var 18..265 : x25;
var 57..202 : x26;
var 6..122 : x27;
var 65..276 : x28;
var 59..80 : x29;
var 26..77 : x30;
var 15..108 : x31;
var 31..177 : x32;
var 23..67 : x33;
var 81..131 : x34;
var 52..251 : x35;
var 9..235 : x36;
var 14..142 : x37;
var 78..146 : x38;
var 44..224 : x39;
array [int,int] of 3..281 : t0=[|67,121,135,62,90,107,67,83,104,93,157,15,114,135,88,180,197,86,130,125,124,240,100,50,80|
103,88,154,59,78,193,117,55,96,64,195,78,109,90,31,114,110,129,113,29,108,127,67,44,72|
85,129,113,54,255,68,159,103,68,69,205,55,120,146,118,174,218,99,127,142,39,200,84,19,65|
121,84,51,51,109,154,153,73,119,99,10,31,99,112,74,121,167,92,232,30,113,197,102,17,79|
130,114,167,53,119,226,94,64,77,116,145,68,79,148,61,176,274,141,244,140,67,153,109,35,79|
84,97,123,55,154,222,71,86,121,78,206,19,142,63,55,194,209,87,251,85,155,126,69,51,67|
114,86,155,53,148,53,157,141,102,77,212,59,118,78,49,80,226,81,212,46,67,135,96,58,77|
58,101,87,53,154,150,51,176,106,69,167,78,135,184,40,139,94,118,229,14,67,242,98,60,75|
68,77,35,52,134,223,82,50,111,118,66,60,107,124,29,112,240,120,123,134,42,246,103,23,60|
84,82,14,52,125,173,115,163,83,58,107,30,109,78,103,122,132,135,88,93,134,138,93,49,66|
125,128,158,61,76,121,152,183,109,103,209,77,137,69,102,167,251,81,136,17,153,95,94,35,80|
93,122,121,59,88,52,99,127,89,91,230,36,138,148,92,109,270,91,207,62,64,152,89,58,61|
81,79,143,61,176,209,118,63,139,78,99,82,81,86,111,76,193,134,148,56,91,96,102,52,74|
115,102,105,57,222,83,155,193,100,104,99,84,96,100,111,155,255,131,118,81,117,134,87,20,75|
136,112,120,55,163,223,123,207,105,80,154,53,83,126,85,128,166,82,187,140,164,101,105,64,79|
87,110,145,58,144,211,107,202,122,88,22,76,100,115,107,166,121,145,188,51,160,88,85,46,78|
62,109,161,63,262,170,78,113,130,95,170,89,93,104,107,129,200,118,189,57,72,122,67,31,70|
55,135,36,63,88,113,52,42,117,104,206,98,125,57,66,129,138,134,90,65,62,146,101,50,70|
124,87,108,62,116,163,114,131,95,61,17,51,101,81,59,74,212,83,128,124,40,231,96,38,74|
59,85,115,50,178,115,142,150,62,58,219,100,129,81,88,178,175,87,168,101,48,177,70,25,60|
122,128,169,52,264,196,101,148,130,104,217,54,95,143,38,132,157,114,249,77,80,248,100,31,64|
139,131,123,60,189,85,156,126,124,85,185,77,116,90,85,180,212,141,181,110,87,166,85,16,71|
106,80,151,62,245,203,74,179,68,74,166,90,81,65,99,76,223,96,171,72,120,236,79,60,79|
142,121,24,62,67,57,40,105,102,93,204,40,92,107,82,190,207,110,239,42,96,142,73,25,69|
57,131,136,60,201,115,56,121,109,64,68,65,107,149,114,181,245,84,219,48,130,131,92,36,80|
112,132,159,58,249,178,53,108,108,100,18,71,141,148,24,129,108,138,116,115,102,175,91,52,60|
135,82,36,59,195,124,155,206,97,87,50,88,125,115,33,130,263,88,98,71,154,157,90,18,68|
131,102,17,57,80,67,124,58,62,80,210,47,145,115,66,179,186,84,125,66,142,88,83,24,77|
72,127,123,62,219,47,104,109,74,72,35,74,91,114,82,128,169,96,199,142,85,231,71,38,80|
102,116,96,50,91,63,120,94,99,99,52,48,95,117,12,100,204,86,200,93,97,142,93,50,70|
115,123,62,62,260,50,107,55,138,124,229,103,91,126,118,165,138,116,178,32,68,176,101,48,80|
81,84,168,51,132,162,111,133,65,89,113,101,142,167,52,190,109,82,188,140,111,89,70,48,74|
108,109,54,57,142,214,142,104,123,116,145,39,91,158,49,124,245,130,205,30,59,101,80,46,77|
136,112,156,63,206,52,149,144,94,116,187,17,133,160,57,131,224,131,118,18,67,212,108,59,74|
84,92,31,62,272,150,106,181,124,88,71,21,134,144,34,147,138,77,222,19,90,163,90,40,66|
124,128,69,57,188,82,51,88,76,108,192,54,94,74,115,181,204,89,172,32,110,247,109,40,61|
137,97,171,57,72,121,70,81,62,113,65,89,119,64,78,166,140,117,212,94,62,191,74,17,74|
96,93,28,51,76,189,140,195,122,102,78,69,126,61,35,137,147,88,164,33,72,210,102,20,68|
88,110,15,51,106,88,58,144,121,91,86,49,100,140,28,142,148,135,155,119,142,207,98,33,75|
70,138,23,50,153,49,75,137,107,61,42,18,114,148,117,163,198,82,242,16,94,221,91,54,62|
55,86,91,60,219,78,101,92,120,96,65,39,108,55,87,160,166,144,162,14,57,252,75,42,62|
102,85,122,52,251,127,47,68,77,71,230,88,137,86,110,151,180,114,149,73,112,110,87,55,66|
127,132,32,58,150,111,97,77,77,69,19,42,78,178,37,202,195,121,116,124,140,104,70,20,73|
139,80,121,61,136,60,139,70,134,58,175,48,86,182,15,69,228,87,233,111,117,199,92,21,64|
58,103,122,55,257,80,45,151,134,122,217,54,81,63,29,194,257,94,177,48,80,90,96,24,63|
103,120,153,55,218,78,104,177,86,73,37,51,126,113,66,121,222,110,148,120,126,214,104,18,66|
107,119,167,56,126,223,144,54,91,108,106,69,79,168,87,79,264,132,91,104,80,89,77,34,66|
88,97,40,52,200,161,139,202,134,114,147,64,83,180,70,114,94,145,221,24,133,207,79,63,65|
136,105,143,51,231,189,144,86,101,100,167,54,132,90,82,171,159,85,142,65,67,110,74,18,75|
63,90,163,60,67,195,155,136,113,115,134,96,104,169,63,117,168,87,88,56,56,98,76,54,71|
70,120,109,63,146,58,67,168,71,76,116,32,130,92,43,123,115,90,182,91,156,122,79,39,69|
79,116,116,59,228,57,46,85,121,88,234,26,99,73,84,100,100,114,133,122,157,167,92,63,64|
63,106,96,60,180,133,104,42,120,68,16,50,138,184,54,145,255,73,197,114,74,232,103,39,69|
64,81,120,61,129,68,61,174,87,64,37,72,128,164,16,68,196,130,224,31,58,167,108,25,80|
88,125,51,56,145,59,158,75,122,73,31,66,132,110,114,196,191,73,101,101,73,233,74,46,67|
92,96,23,53,158,174,85,68,57,112,87,46,82,146,58,98,213,89,200,140,99,198,86,52,75|
71,94,22,55,234,69,97,114,84,62,131,65,92,128,48,80,274,110,166,69,104,136,68,18,77|
102,132,143,56,131,195,67,144,69,78,219,64,145,79,16,69,246,125,191,131,100,158,76,14,80|
72,85,103,54,186,197,110,79,64,87,53,106,93,164,110,151,86,124,98,47,141,86,109,43,74|
103,118,82,57,252,62,89,149,58,87,172,52,88,107,24,141,209,117,216,52,149,199,83,31,62|
68,87,66,54,180,134,83,70,138,64,230,17,143,159,106,113,191,100,103,108,90,98,100,42,62|
56,79,28,63,124,150,60,182,69,69,227,48,84,135,31,143,155,135,90,28,100,174,94,51,71|
77,92,78,59,161,173,138,147,102,60,39,21,106,59,100,85,154,115,159,126,125,167,84,25,64|
142,120,54,57,219,132,104,70,103,93,31,67,116,112,61,141,137,99,210,57,151,148,106,62,73|
89,130,71,62,125,207,106,64,134,110,47,35,99,129,104,126,156,85,134,119,83,250,76,50,78|
126,104,69,60,90,72,54,170,79,103,48,84,137,165,13,59,170,128,101,93,153,115,94,25,79|
82,89,75,54,208,223,71,177,114,90,54,43,116,134,68,172,276,126,162,102,130,156,87,47,59|
125,134,37,62,154,221,155,171,71,114,65,53,79,141,46,183,157,87,126,45,47,149,108,18,80|
108,132,169,54,87,178,135,156,84,93,42,26,144,59,98,193,151,101,209,136,108,231,97,55,71|
119,104,152,54,209,154,134,72,120,119,198,99,114,131,43,62,225,141,222,33,136,129,105,58,66|
123,107,121,62,243,178,61,83,71,92,200,56,92,159,46,105,196,117,195,78,51,214,92,63,78|
88,108,139,59,68,86,109,93,78,64,159,98,139,143,25,75,195,123,152,86,49,148,102,37,61|
119,99,42,53,166,110,72,85,117,59,137,92,122,128,53,126,195,114,151,15,157,154,109,46,76|
101,83,42,59,126,132,129,168,98,107,194,59,120,168,120,97,251,92,131,72,92,221,90,46,75|
98,112,128,59,102,223,52,76,93,95,119,86,146,110,89,119,131,88,251,126,90,245,97,20,74|
54,138,171,56,115,175,59,83,87,80,42,104,114,80,24,89,262,83,225,128,130,105,76,15,80|
61,87,61,61,257,175,57,77,59,61,56,55,129,116,10,71,126,86,182,42,101,203,106,31,67|
74,98,66,53,127,56,92,132,74,92,18,85,94,64,73,110,113,126,96,65,41,212,97,32,59|
70,98,65,56,233,222,73,50,106,72,71,91,94,176,74,121,99,85,147,120,143,142,91,53,72|
58,87,37,50,144,87,153,140,101,74,144,44,78,88,24,180,254,84,149,25,40,93,76,60,59|
109,83,72,61,251,181,55,112,65,118,44,96,134,126,12,83,244,77,128,46,100,194,77,57,59|
90,103,69,52,271,168,130,89,69,111,10,16,129,85,53,144,186,85,215,22,80,239,99,46,75|
107,105,129,63,187,51,104,102,99,68,132,19,86,95,65,195,204,76,126,16,68,195,92,18,76|
137,101,108,59,220,122,62,198,88,72,176,50,96,108,117,178,103,78,160,92,70,96,71,38,59|
67,92,91,56,119,103,45,70,73,105,61,73,138,163,60,78,148,77,215,100,148,91,86,46,59|
123,114,164,50,252,67,124,59,130,81,233,31,119,163,122,190,169,118,221,46,133,93,68,20,68|
102,112,35,59,109,172,130,66,96,93,72,33,118,171,76,161,263,84,135,91,141,79,95,22,68|
127,136,96,58,203,147,89,148,70,90,64,79,121,157,95,112,206,109,135,22,66,192,79,25,67|
119,118,16,63,225,93,57,84,96,110,54,19,121,172,30,92,267,80,155,124,71,194,91,61,72|
98,133,80,51,248,112,77,187,83,73,41,108,99,157,33,131,234,133,203,82,66,227,107,59,61|
124,135,111,54,71,139,145,69,63,69,65,65,131,140,85,144,173,84,115,102,164,192,96,35,74|
55,135,133,53,240,62,43,181,137,80,130,47,120,173,24,95,153,141,185,112,43,140,71,21,62|
102,80,129,60,211,194,143,60,108,96,91,18,95,172,40,186,132,74,221,111,72,129,68,16,64|
107,100,141,53,107,137,130,63,70,89,215,79,132,152,71,165,275,118,231,135,38,234,103,19,75|
145,89,109,61,274,130,150,103,131,96,226,88,81,103,24,153,261,107,137,58,102,155,101,59,80|
115,95,92,58,180,115,156,173,89,79,97,106,100,177,10,185,100,96,152,117,129,180,77,62,64|
68,123,70,57,119,65,132,116,60,81,157,85,106,143,48,80,216,121,100,121,59,216,87,56,71|
130,114,90,59,197,116,77,208,96,78,156,105,78,101,121,144,139,137,168,81,112,212,92,64,65|
131,94,48,63,80,79,106,94,123,114,203,49,109,66,111,144,103,139,129,73,149,193,75,45,64|
110,103,94,63,159,223,150,128,115,97,232,98,78,147,92,193,229,111,135,97,130,124,74,31,74|
69,84,82,63,227,127,51,105,134,117,128,30,84,143,82,172,123,98,226,86,145,203,100,34,62|
79,129,80,63,132,72,58,103,110,107,40,16,143,132,58,95,209,77,116,29,44,92,91,21,66|
143,127,63,62,204,123,131,103,127,124,25,63,117,183,81,143,262,117,245,127,119,87,100,16,80|
115,133,28,61,145,154,87,154,127,90,202,42,120,183,8,127,89,112,140,25,64,105,86,27,77|
72,110,52,59,68,107,43,45,60,113,30,64,90,144,98,193,229,83,98,41,54,159,105,60,65|
145,95,150,57,271,75,97,78,116,117,90,103,91,119,61,140,179,77,104,122,154,82,73,59,80|
121,127,136,54,259,187,149,102,93,111,53,69,106,68,17,200,110,107,238,84,40,164,67,44,68|
139,131,58,61,78,47,86,73,95,124,199,47,129,124,96,80,163,76,140,46,111,198,103,23,59|
130,81,29,60,160,77,107,152,139,77,230,90,112,160,90,63,183,135,93,87,68,140,109,23,62|
94,123,47,55,275,187,117,137,86,76,137,64,79,146,49,120,251,108,249,114,59,154,79,25,60|
87,97,149,54,85,62,147,163,75,84,230,18,117,127,87,161,155,136,187,78,150,94,105,14,76|
76,110,39,60,122,183,134,63,67,89,113,65,79,129,36,115,202,103,200,55,130,226,98,25,73|
136,126,35,54,190,203,55,126,80,83,123,59,116,55,77,122,104,95,113,40,78,207,95,18,78|
107,125,110,60,208,198,115,78,125,75,69,80,91,128,27,150,143,131,133,36,113,136,109,55,63|
118,114,21,55,190,83,129,122,138,81,48,59,96,98,23,124,193,113,158,113,61,254,76,33,68|
77,82,32,56,81,176,105,60,66,76,101,53,122,112,87,61,275,79,224,29,64,235,93,60,69|
132,125,54,58,211,107,95,195,68,58,134,15,94,87,120,64,176,132,172,62,73,220,87,50,72|
70,139,137,51,137,160,129,198,92,89,89,72,142,141,52,88,89,77,160,31,149,84,101,35,65|
99,90,150,60,144,111,147,81,82,105,21,52,95,63,7,183,170,88,177,136,137,224,104,37,62|
117,113,94,50,261,92,49,151,127,113,225,45,85,114,22,150,84,116,151,28,138,125,81,19,61|
73,125,139,57,173,143,57,159,62,65,88,41,109,105,88,96,88,115,247,57,154,114,88,28,70|
114,116,159,53,232,139,100,67,121,115,220,39,105,85,33,108,150,136,213,58,85,83,74,34,61|
54,93,87,50,152,182,63,142,67,78,137,92,92,171,55,69,222,132,243,91,134,255,86,17,62|
87,107,171,62,260,49,50,163,70,76,139,102,84,165,77,198,244,110,223,38,81,139,90,23,72|
95,103,29,50,216,193,97,54,57,70,196,50,102,59,67,62,232,135,205,37,156,218,82,42,63|
94,79,59,63,88,228,110,178,122,81,72,68,118,55,120,75,122,104,243,108,76,212,102,24,68|
74,117,48,57,112,90,157,163,132,84,84,56,144,83,121,107,273,134,149,89,121,220,109,62,66|
111,94,29,63,126,145,108,71,95,86,155,80,131,79,26,107,250,88,129,45,102,232,88,56,60|
61,99,158,60,148,217,77,164,122,62,185,87,109,71,65,138,128,86,249,21,89,213,89,21,59|
122,97,107,57,164,117,78,80,58,66,70,72,122,72,73,101,217,74,232,59,72,119,82,24,66|
104,130,51,59,265,221,64,72,75,61,126,103,122,138,49,118,226,85,118,20,151,159,90,56,67|
145,118,125,50,117,112,124,184,59,103,195,66,101,58,122,100,246,118,92,139,43,236,67,29,80|
138,120,133,53,71,90,117,156,94,68,213,71,102,153,25,99,121,81,165,70,96,178,104,42,71|
99,110,37,52,82,194,133,63,125,79,132,37,98,73,10,181,190,142,121,130,85,151,100,55,72|
92,127,105,56,130,149,125,120,117,72,231,69,79,85,114,68,97,94,236,55,55,139,91,34,64|
136,103,88,54,185,229,67,98,100,108,52,76,120,139,83,101,201,110,111,71,147,104,81,50,76|
143,100,26,55,152,94,128,42,68,65,49,74,97,93,107,93,115,89,226,129,110,113,94,40,68|
118,111,97,57,259,87,128,128,76,116,226,91,83,115,115,158,210,97,250,123,79,251,92,23,59|
75,87,69,58,273,229,81,209,75,69,9,95,86,60,75,131,132,125,176,132,49,241,91,31,63|
102,108,45,61,215,206,111,63,72,99,68,51,106,117,36,153,191,98,225,29,79,239,70,17,64|
122,110,61,60,273,174,75,178,121,106,97,93,101,118,26,120,139,89,232,102,120,256,102,34,63|
64,93,146,54,263,139,76,194,93,96,207,23,98,82,17,92,245,95,144,28,94,91,105,23,72|
68,105,162,50,107,74,161,142,102,89,115,59,126,70,43,98,121,104,213,113,133,96,106,14,70|
125,140,156,63,227,95,58,74,66,77,135,57,120,166,98,183,118,124,146,110,118,201,96,61,77|
120,98,110,58,266,75,127,121,63,105,159,103,97,129,100,202,235,138,255,42,58,96,99,21,80|
119,116,90,63,223,190,108,123,121,77,208,79,135,177,117,94,166,95,217,71,129,91,90,53,68|
78,81,117,62,179,211,81,47,137,119,77,106,96,73,83,85,251,145,243,120,154,217,79,35,67|
58,116,80,63,132,220,91,184,77,111,210,57,113,162,95,88,201,123,119,99,85,107,68,35,75|
66,88,115,59,256,163,150,50,91,63,194,16,146,107,12,125,200,77,126,17,45,217,98,43,63|
75,130,30,63,185,204,134,194,117,86,108,81,81,113,47,166,86,100,125,61,141,140,106,38,74|
68,110,45,60,170,228,99,119,84,88,107,89,102,110,29,130,239,83,226,22,61,184,96,33,73|
100,137,22,52,259,151,112,171,135,99,224,45,100,94,111,132,90,78,100,123,75,80,91,57,79|
124,129,136,56,125,180,102,79,73,71,224,50,116,63,85,190,172,87,120,26,73,206,95,42,69|
94,99,159,50,249,118,145,46,64,59,107,75,141,119,112,164,190,97,111,116,47,218,71,16,60|
76,78,68,53,159,66,86,44,131,86,23,32,133,134,79,144,139,76,96,117,147,92,87,35,75|
68,95,119,56,102,68,86,177,96,60,142,50,80,174,13,201,95,114,106,115,40,254,78,16,63|
124,90,149,51,96,229,51,44,108,108,87,60,84,141,57,105,264,141,241,72,74,148,83,21,76|
88,124,100,59,275,94,116,143,102,98,212,21,109,134,66,154,123,138,150,58,81,193,81,40,80|
78,139,48,63,89,171,152,134,74,70,53,70,101,182,46,181,269,103,164,75,163,151,72,23,66|]; 
array [int,int] of SmartTableOp: signs0=[|Int,Int,All,LtInt,Int,Int,LtInt,All,GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,GtInt,Int,GtInt|Int,LtInt,Int,GtInt,Int,GtInt,GtInt,LtInt,Int,All,Int,GtInt,LtInt,GtInt,Int,Int,Int,LtInt,All,Int,Int,Int,Int,GtInt,Int|Int,Int,All,Int,Int,LtInt,GtInt,Int,Int,GtInt,LtInt,Int,LtInt,Int,LtInt,Int,Int,LtInt,Int,LtInt,LtInt,Int,Int,LtInt,GtInt|Int,Int,Int,LtInt,All,LtInt,Int,Int,Int,Int,LtInt,All,GtInt,Int,Int,Int,GtInt,GtInt,Int,Int,Int,GtInt,Int,Int,Int|Int,GtInt,Int,LtInt,Int,All,LtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,Int,GtInt,LtInt,All,Int,LtInt,Int,Int,LtInt|Int,GtInt,Int,GtInt,LtInt,GtInt,Int,LtInt,Int,All,GtInt,GtInt,All,LtInt,GtInt,GtInt,Int,LtInt,Int,GtInt,Int,Int,Int,Int,Int|All,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,LtInt,LtInt,Int,Int,Int,GtInt,LtInt,Int,LtInt,LtInt,LtInt,GtInt,Int,Int,GtInt|Int,Int,Int,All,GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,All,Int,GtInt,GtInt,GtInt,All,Int,All,All,Int,All,GtInt|GtInt,Int,GtInt,GtInt,Int,Int,Int,LtInt,GtInt,Int,Int,Int,LtInt,Int,GtInt,All,Int,All,All,Int,Int,Int,Int,All,GtInt|Int,Int,LtInt,Int,GtInt,GtInt,Int,GtInt,GtInt,LtInt,Int,Int,Int,All,Int,Int,GtInt,LtInt,All,All,Int,Int,GtInt,Int,LtInt|Int,LtInt,LtInt,LtInt,LtInt,All,LtInt,All,GtInt,Int,All,Int,Int,Int,GtInt,GtInt,Int,Int,Int,All,GtInt,Int,Int,GtInt,Int|Int,Int,All,Int,Int,Int,GtInt,All,Int,LtInt,Int,Int,All,Int,Int,GtInt,Int,Int,All,Int,Int,LtInt,GtInt,Int,GtInt|Int,All,All,Int,Int,Int,Int,GtInt,Int,Int,GtInt,All,Int,Int,Int,Int,Int,All,Int,Int,All,Int,Int,Int,LtInt|Int,Int,LtInt,Int,Int,Int,Int,Int,All,Int,LtInt,Int,Int,LtInt,Int,All,Int,LtInt,Int,Int,All,All,GtInt,Int,Int|GtInt,GtInt,Int,Int,Int,Int,GtInt,Int,Int,LtInt,GtInt,Int,Int,Int,Int,All,All,All,Int,All,Int,Int,Int,GtInt,All|Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,All,Int,All,LtInt,Int,Int,GtInt,All,Int,LtInt,Int,LtInt|GtInt,Int,Int,Int,Int,GtInt,All,Int,LtInt,LtInt,GtInt,LtInt,Int,Int,Int,LtInt,Int,Int,Int,LtInt,GtInt,Int,Int,Int,Int|Int,LtInt,GtInt,Int,Int,GtInt,Int,Int,Int,Int,All,GtInt,Int,Int,Int,Int,Int,GtInt,LtInt,LtInt,LtInt,LtInt,Int,Int,LtInt|LtInt,All,Int,Int,Int,Int,LtInt,Int,Int,GtInt,GtInt,Int,GtInt,Int,LtInt,Int,Int,LtInt,Int,All,Int,Int,Int,LtInt,GtInt|Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt,LtInt,Int,Int,Int,LtInt,LtInt,Int,All,Int,GtInt|All,Int,Int,Int,Int,Int,Int,All,Int,LtInt,Int,LtInt,Int,Int,LtInt,All,Int,LtInt,LtInt,Int,LtInt,Int,Int,Int,Int|Int,Int,LtInt,LtInt,LtInt,Int,Int,Int,All,Int,Int,Int,GtInt,Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int|Int,LtInt,GtInt,Int,Int,GtInt,GtInt,GtInt,GtInt,LtInt,LtInt,Int,Int,LtInt,All,Int,Int,Int,Int,Int,Int,GtInt,Int,All,Int|GtInt,LtInt,Int,Int,Int,GtInt,Int,Int,Int,LtInt,All,Int,Int,Int,Int,GtInt,All,Int,Int,Int,Int,LtInt,Int,LtInt,Int|GtInt,All,LtInt,GtInt,GtInt,GtInt,LtInt,Int,Int,GtInt,Int,Int,GtInt,Int,Int,GtInt,GtInt,Int,All,GtInt,All,Int,Int,GtInt,Int|Int,LtInt,LtInt,Int,GtInt,GtInt,Int,Int,GtInt,All,Int,Int,Int,GtInt,GtInt,Int,GtInt,Int,Int,All,Int,All,LtInt,All,Int|GtInt,LtInt,Int,Int,Int,GtInt,All,Int,Int,Int,LtInt,Int,Int,LtInt,Int,Int,LtInt,LtInt,All,All,Int,Int,LtInt,GtInt,Int|LtInt,Int,All,All,GtInt,Int,Int,Int,All,Int,Int,GtInt,LtInt,GtInt,GtInt,All,Int,All,Int,Int,GtInt,Int,Int,Int,Int|Int,Int,LtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,LtInt,Int,Int,GtInt,Int,Int,Int,Int,GtInt,LtInt|GtInt,Int,Int,Int,Int,All,LtInt,Int,Int,GtInt,All,LtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,LtInt,Int,Int,GtInt|LtInt,LtInt,LtInt,All,Int,All,Int,GtInt,LtInt,Int,Int,GtInt,Int,Int,All,Int,All,LtInt,LtInt,LtInt,All,Int,Int,Int,All|Int,Int,Int,Int,GtInt,Int,LtInt,Int,LtInt,Int,Int,Int,LtInt,LtInt,LtInt,Int,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int|GtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,LtInt,GtInt,LtInt,Int,LtInt,Int,GtInt|All,Int,Int,GtInt,Int,LtInt,GtInt,Int,Int,Int,Int,Int,All,LtInt,Int,Int,Int,Int,GtInt,Int,GtInt,All,GtInt,Int,Int|Int,GtInt,GtInt,GtInt,GtInt,Int,Int,LtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,All|Int,GtInt,GtInt,Int,Int,GtInt,Int,Int,LtInt,Int,Int,All,All,Int,All,Int,LtInt,All,Int,Int,Int,LtInt,Int,LtInt,GtInt|LtInt,Int,Int,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,All,Int,GtInt,Int,Int,Int|All,Int,Int,LtInt,GtInt,Int,Int,Int,All,Int,Int,LtInt,Int,Int,All,Int,LtInt,Int,Int,Int,Int,Int,Int,LtInt,Int|All,Int,Int,GtInt,LtInt,Int,LtInt,Int,Int,GtInt,GtInt,Int,LtInt,Int,Int,Int,Int,GtInt,GtInt,All,Int,Int,Int,Int,Int|LtInt,GtInt,GtInt,All,All,GtInt,Int,Int,GtInt,Int,Int,Int,LtInt,All,Int,Int,Int,Int,Int,Int,GtInt,GtInt,Int,Int,Int|GtInt,Int,Int,LtInt,LtInt,GtInt,GtInt,Int,LtInt,GtInt,Int,Int,Int,Int,Int,All,Int,GtInt,Int,LtInt,Int,Int,LtInt,Int,Int|Int,Int,Int,Int,Int,GtInt,GtInt,Int,GtInt,Int,GtInt,GtInt,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,GtInt,GtInt|Int,GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,All,Int,GtInt,Int,LtInt,GtInt,Int,GtInt,Int,Int,GtInt,Int|Int,Int,GtInt,Int,Int,Int,All,Int,Int,LtInt,LtInt,Int,LtInt,All,Int,Int,Int,Int,LtInt,GtInt,Int,Int,All,Int,LtInt|Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,LtInt,Int,Int,Int,LtInt,GtInt,Int,Int|Int,Int,All,Int,Int,Int,Int,Int,LtInt,LtInt,Int,GtInt,GtInt,Int,Int,LtInt,All,GtInt,GtInt,Int,All,Int,Int,Int,LtInt|Int,All,LtInt,Int,All,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,GtInt,Int,LtInt,Int,LtInt,LtInt,Int,Int,Int,Int,Int|Int,Int,GtInt,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,All,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int|Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,GtInt,All,LtInt,Int,Int,Int,Int,Int,Int,Int|Int,GtInt,Int,Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,Int,Int,Int,GtInt,Int,All,Int,Int,All,LtInt,Int,Int,Int,Int,All,LtInt,Int,LtInt,LtInt,Int,GtInt,All,All,GtInt|LtInt,Int,Int,GtInt,All,All,Int,Int,Int,Int,Int,GtInt,GtInt,Int,Int,GtInt,Int,GtInt,Int,Int,Int,Int,LtInt,LtInt,Int|Int,All,LtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int|Int,Int,Int,Int,All,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,GtInt,Int,GtInt,LtInt,All,Int,All,GtInt,LtInt|Int,All,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,LtInt,LtInt,Int,Int,Int,GtInt,Int,GtInt,GtInt|All,Int,Int,All,All,All,Int,LtInt,LtInt,All,Int,Int,Int,Int,LtInt,Int,Int,All,All,All,Int,All,Int,Int,LtInt|GtInt,GtInt,Int,Int,LtInt,LtInt,Int,All,GtInt,Int,GtInt,Int,Int,Int,Int,All,LtInt,Int,GtInt,LtInt,All,GtInt,GtInt,LtInt,Int|All,Int,Int,LtInt,All,LtInt,Int,LtInt,LtInt,Int,LtInt,All,Int,Int,Int,GtInt,Int,Int,Int,GtInt,All,GtInt,Int,Int,All|All,Int,LtInt,Int,LtInt,All,Int,Int,GtInt,LtInt,GtInt,Int,Int,Int,Int,Int,All,LtInt,GtInt,Int,Int,Int,Int,Int,Int|Int,Int,GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,LtInt,Int,Int,LtInt,GtInt,GtInt,GtInt,GtInt,LtInt,Int,Int|Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt,Int,GtInt,Int,LtInt,Int,GtInt,LtInt,LtInt,Int|Int,LtInt,Int,Int,Int,LtInt,GtInt,All,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,GtInt,Int,GtInt,GtInt,Int,LtInt,Int|Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,GtInt,Int,All,Int,Int,All|Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt,GtInt,Int,Int,GtInt,LtInt,Int,Int,GtInt,Int,GtInt|Int,LtInt,Int,LtInt,Int,All,Int,All,LtInt,Int,All,GtInt,Int,Int,Int,GtInt,Int,Int,Int,GtInt,Int,LtInt,Int,All,All|All,Int,Int,All,Int,Int,All,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,GtInt,GtInt,LtInt|Int,LtInt,All,Int,Int,Int,GtInt,Int,Int,LtInt,LtInt,Int,All,Int,GtInt,All,Int,Int,GtInt,Int,Int,Int,Int,Int,Int|LtInt,Int,Int,Int,GtInt,Int,Int,LtInt,Int,Int,LtInt,All,GtInt,GtInt,All,LtInt,GtInt,Int,Int,GtInt,Int,GtInt,Int,Int,Int|Int,GtInt,GtInt,Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,All,Int,All,Int,Int,Int,Int,LtInt,All,All,Int,Int|Int,Int,Int,Int,GtInt,Int,All,All,All,Int,LtInt,Int,All,All,Int,All,GtInt,Int,Int,LtInt,Int,Int,GtInt,Int,Int|LtInt,Int,Int,All,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,All,All,Int,GtInt,Int,Int,Int,Int,Int|Int,Int,Int,Int,Int,Int,All,LtInt,Int,All,GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,All,Int,LtInt,Int,Int,Int|Int,Int,Int,LtInt,Int,LtInt,Int,Int,Int,Int,GtInt,GtInt,GtInt,All,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,Int,Int,Int,Int,LtInt,All,Int,Int,Int,LtInt,All,Int,LtInt,Int,Int,Int,GtInt,Int,GtInt,LtInt,LtInt,Int,GtInt,Int|GtInt,LtInt,Int,Int,Int,Int,All,GtInt,Int,Int,GtInt,LtInt,Int,LtInt,Int,Int,All,Int,LtInt,All,Int,Int,All,All,Int|Int,LtInt,All,All,GtInt,GtInt,LtInt,LtInt,Int,Int,Int,GtInt,GtInt,Int,Int,All,Int,GtInt,LtInt,GtInt,LtInt,Int,Int,GtInt,Int|Int,Int,Int,Int,LtInt,GtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,Int,All,Int,LtInt,Int,Int,Int|Int,LtInt,Int,Int,LtInt,Int,GtInt,LtInt,LtInt,Int,Int,LtInt,Int,LtInt,Int,All,Int,GtInt,GtInt,Int,Int,GtInt,LtInt,All,LtInt|Int,All,Int,Int,Int,LtInt,LtInt,Int,Int,LtInt,Int,All,Int,Int,GtInt,Int,Int,GtInt,GtInt,Int,Int,Int,All,Int,Int|All,Int,Int,Int,All,Int,All,Int,Int,LtInt,LtInt,Int,GtInt,Int,Int,LtInt,Int,All,Int,Int,Int,GtInt,Int,Int,GtInt|GtInt,Int,All,Int,Int,GtInt,LtInt,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,GtInt,LtInt,Int,All,Int,Int,Int,LtInt|Int,Int,GtInt,Int,Int,All,Int,Int,LtInt,Int,Int,Int,LtInt,Int,LtInt,Int,Int,GtInt,GtInt,Int,Int,All,GtInt,Int,GtInt|LtInt,LtInt,Int,All,Int,LtInt,LtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,GtInt,Int,Int,LtInt|Int,LtInt,Int,All,Int,Int,LtInt,All,Int,Int,Int,LtInt,Int,Int,Int,LtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int,Int|Int,Int,Int,All,GtInt,Int,LtInt,Int,All,LtInt,Int,Int,LtInt,LtInt,All,Int,Int,LtInt,GtInt,LtInt,Int,All,LtInt,Int,All|Int,Int,LtInt,Int,LtInt,Int,Int,LtInt,LtInt,All,Int,All,LtInt,Int,Int,LtInt,GtInt,Int,Int,Int,Int,Int,All,Int,Int|Int,Int,Int,Int,Int,Int,Int,Int,All,Int,Int,LtInt,LtInt,Int,Int,All,GtInt,GtInt,Int,Int,Int,Int,Int,Int,Int|Int,GtInt,All,Int,Int,Int,LtInt,LtInt,Int,GtInt,Int,LtInt,LtInt,Int,Int,Int,GtInt,Int,Int,GtInt,Int,Int,GtInt,Int,Int|Int,Int,LtInt,Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,All,All,Int,Int,Int,Int|Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,GtInt,GtInt,Int,Int,Int,Int,All,GtInt,LtInt,Int,Int,Int,Int|GtInt,GtInt,All,LtInt,Int,Int,Int,Int,GtInt,Int,LtInt,Int,Int,Int,Int,All,GtInt,GtInt,Int,All,LtInt,Int,LtInt,Int,GtInt|All,Int,All,LtInt,All,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt,LtInt,GtInt,GtInt,Int,Int,Int,GtInt,GtInt|LtInt,Int,GtInt,GtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,All,Int,Int,Int,Int,GtInt,Int,Int,Int,GtInt,Int|Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,GtInt,All,Int,Int,Int,GtInt,Int,All,Int,Int,GtInt,GtInt,Int,Int,Int,Int|Int,All,GtInt,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,LtInt,GtInt,Int,Int,All,Int,LtInt,GtInt,All,Int,Int,Int|GtInt,Int,Int,LtInt,LtInt,Int,Int,GtInt,Int,Int,Int,Int,GtInt,All,Int,GtInt,Int,LtInt,Int,LtInt,Int,LtInt,GtInt,GtInt,Int|Int,Int,Int,All,Int,Int,LtInt,Int,Int,Int,Int,Int,GtInt,GtInt,Int,GtInt,Int,Int,All,Int,Int,Int,Int,Int,Int|LtInt,Int,Int,Int,Int,GtInt,GtInt,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt,Int,All,LtInt,GtInt,GtInt,Int|LtInt,Int,Int,Int,Int,Int,Int,GtInt,All,Int,Int,GtInt,LtInt,Int,Int,All,GtInt,GtInt,Int,Int,Int,GtInt,Int,Int,LtInt|Int,Int,Int,Int,All,Int,GtInt,Int,LtInt,LtInt,GtInt,GtInt,Int,LtInt,GtInt,Int,Int,LtInt,Int,LtInt,All,Int,GtInt,All,Int|Int,Int,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,All,LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,All,Int,GtInt|Int,Int,Int,Int,Int,Int,LtInt,Int,LtInt,Int,Int,Int,LtInt,GtInt,Int,Int,Int,Int,Int,Int,GtInt,LtInt,GtInt,LtInt,GtInt|Int,GtInt,Int,GtInt,Int,All,Int,Int,Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,All,GtInt,Int,GtInt,Int|All,Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int,GtInt,LtInt,LtInt,Int,GtInt,LtInt,Int,Int,Int,GtInt,LtInt,Int,Int,Int,GtInt|GtInt,All,Int,GtInt,GtInt,LtInt,Int,Int,Int,Int,All,Int,GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int|LtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,All,Int,Int,GtInt,Int,LtInt,Int,Int,Int,Int,All,LtInt,Int,GtInt|LtInt,Int,GtInt,Int,Int,LtInt,GtInt,Int,GtInt,Int,All,Int,Int,Int,LtInt,GtInt,Int,Int,GtInt,GtInt,GtInt,Int,Int,GtInt,Int|All,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,All,GtInt,Int,Int,LtInt,Int,Int,LtInt,GtInt,LtInt,LtInt,Int,All,All,Int|Int,Int,Int,Int,Int,Int,All,GtInt,LtInt,Int,Int,Int,All,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,LtInt,Int,Int|Int,Int,LtInt,Int,All,Int,Int,Int,LtInt,LtInt,Int,Int,GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt|LtInt,GtInt,Int,Int,Int,All,Int,Int,GtInt,GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,All,All,All,Int,Int,All|Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,GtInt,Int,Int,GtInt,GtInt,Int,LtInt,Int,Int,Int,LtInt|Int,LtInt,All,LtInt,GtInt,LtInt,All,Int,Int,Int,GtInt,Int,GtInt,GtInt,Int,All,Int,GtInt,Int,Int,Int,Int,Int,Int,Int|LtInt,Int,All,LtInt,LtInt,Int,Int,Int,LtInt,GtInt,Int,Int,LtInt,Int,GtInt,Int,LtInt,Int,Int,Int,Int,GtInt,LtInt,LtInt,Int|GtInt,Int,Int,Int,Int,LtInt,Int,Int,GtInt,All,All,Int,Int,Int,Int,Int,LtInt,GtInt,GtInt,All,Int,LtInt,LtInt,LtInt,Int|Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,LtInt,LtInt,Int,LtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt|All,LtInt,Int,GtInt,Int,Int,Int,All,All,LtInt,Int,Int,Int,Int,GtInt,Int,All,All,Int,Int,Int,Int,Int,Int,Int|LtInt,Int,Int,Int,All,GtInt,Int,LtInt,All,Int,GtInt,LtInt,GtInt,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,GtInt,LtInt|Int,GtInt,LtInt,LtInt,Int,Int,LtInt,Int,All,Int,All,Int,GtInt,Int,Int,Int,All,Int,Int,Int,Int,Int,Int,Int,Int|Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,LtInt,All,Int,Int,All,GtInt,Int,Int,LtInt,Int,Int,Int,LtInt,Int,All|All,LtInt,Int,Int,Int,Int,Int,GtInt,Int,All,Int,GtInt,Int,LtInt,LtInt,LtInt,Int,Int,Int,Int,Int,All,Int,Int,LtInt|Int,LtInt,Int,LtInt,Int,Int,LtInt,Int,All,GtInt,Int,Int,LtInt,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int|Int,Int,GtInt,Int,Int,Int,Int,GtInt,GtInt,Int,GtInt,Int,All,LtInt,Int,Int,GtInt,GtInt,Int,Int,Int,Int,All,Int,Int|Int,Int,GtInt,GtInt,GtInt,GtInt,LtInt,LtInt,Int,GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int|Int,GtInt,Int,Int,GtInt,LtInt,Int,LtInt,LtInt,LtInt,Int,Int,GtInt,Int,Int,Int,All,All,Int,LtInt,LtInt,Int,All,LtInt,GtInt|Int,Int,Int,LtInt,LtInt,Int,GtInt,Int,LtInt,Int,GtInt,LtInt,GtInt,LtInt,All,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int|Int,Int,GtInt,LtInt,Int,Int,Int,LtInt,Int,Int,All,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,LtInt,Int,Int,Int,All|Int,LtInt,All,Int,Int,All,LtInt,Int,GtInt,Int,GtInt,Int,Int,GtInt,Int,Int,GtInt,LtInt,Int,Int,Int,GtInt,Int,GtInt,All|All,LtInt,Int,Int,Int,All,LtInt,Int,Int,GtInt,LtInt,LtInt,Int,All,Int,LtInt,GtInt,Int,Int,All,All,All,Int,Int,Int|LtInt,Int,All,Int,GtInt,Int,Int,GtInt,LtInt,Int,LtInt,Int,Int,LtInt,Int,All,LtInt,Int,Int,GtInt,All,GtInt,Int,Int,GtInt|Int,All,All,LtInt,Int,LtInt,GtInt,Int,Int,Int,LtInt,LtInt,All,Int,GtInt,LtInt,Int,GtInt,Int,LtInt,Int,Int,LtInt,Int,Int|LtInt,GtInt,Int,Int,LtInt,All,Int,Int,Int,Int,GtInt,All,All,Int,GtInt,Int,Int,All,Int,Int,Int,Int,Int,Int,Int|Int,Int,LtInt,Int,GtInt,Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,Int,All,Int,Int,Int,LtInt,Int,All,LtInt,LtInt,Int,Int|Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,All,Int,Int,Int,Int,LtInt,All,Int,LtInt,LtInt,All,Int,LtInt,All|Int,Int,Int,Int,GtInt,Int,LtInt,LtInt,Int,Int,LtInt,All,Int,All,Int,Int,Int,Int,Int,Int,LtInt,GtInt,GtInt,GtInt,Int|Int,Int,GtInt,Int,Int,LtInt,LtInt,Int,Int,Int,GtInt,Int,LtInt,Int,Int,All,Int,Int,Int,Int,Int,All,All,LtInt,All|Int,Int,All,GtInt,Int,GtInt,Int,All,GtInt,Int,Int,Int,Int,All,Int,Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int|LtInt,All,LtInt,Int,Int,Int,Int,Int,Int,GtInt,LtInt,GtInt,Int,Int,Int,LtInt,GtInt,GtInt,GtInt,LtInt,LtInt,LtInt,Int,All,Int|Int,Int,Int,Int,All,Int,All,GtInt,Int,All,Int,LtInt,GtInt,Int,Int,GtInt,Int,Int,Int,All,All,Int,LtInt,Int,Int|Int,Int,Int,LtInt,LtInt,Int,Int,Int,LtInt,Int,LtInt,GtInt,LtInt,Int,Int,Int,Int,Int,All,Int,Int,LtInt,Int,GtInt,Int|All,All,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,LtInt,Int,Int,Int,GtInt,All,Int,GtInt,Int,All,LtInt,Int|Int,All,Int,GtInt,All,Int,LtInt,Int,Int,Int,Int,All,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt|Int,LtInt,Int,LtInt,Int,All,GtInt,Int,Int,Int,All,LtInt,Int,LtInt,Int,Int,Int,Int,Int,GtInt,All,GtInt,GtInt,All,Int|All,LtInt,LtInt,Int,Int,LtInt,GtInt,All,Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,GtInt,GtInt,Int,Int|Int,LtInt,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt|Int,All,Int,GtInt,LtInt,Int,GtInt,All,Int,LtInt,All,Int,Int,LtInt,All,LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int|Int,GtInt,Int,LtInt,Int,GtInt,GtInt,LtInt,Int,GtInt,All,Int,LtInt,Int,Int,Int,LtInt,Int,LtInt,GtInt,GtInt,GtInt,Int,Int,Int|All,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,LtInt,Int,Int,Int,Int,Int|Int,Int,LtInt,GtInt,GtInt,Int,LtInt,All,Int,Int,LtInt,Int,GtInt,GtInt,LtInt,Int,Int,GtInt,Int,Int,Int,LtInt,GtInt,Int,Int|Int,Int,Int,Int,Int,LtInt,LtInt,All,GtInt,All,All,GtInt,GtInt,Int,GtInt,Int,Int,Int,All,Int,LtInt,Int,All,Int,LtInt|Int,Int,Int,Int,Int,Int,Int,All,GtInt,Int,Int,Int,All,Int,Int,GtInt,Int,All,Int,Int,Int,Int,All,Int,All|Int,Int,LtInt,Int,LtInt,Int,LtInt,Int,All,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,Int,Int,Int,LtInt,All,LtInt|All,Int,Int,Int,LtInt,Int,All,LtInt,Int,LtInt,Int,Int,Int,Int,LtInt,LtInt,All,GtInt,LtInt,Int,LtInt,Int,Int,Int,Int|Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,All,LtInt,Int,Int,LtInt,GtInt,LtInt,GtInt,GtInt,GtInt|Int,GtInt,Int,Int,LtInt,Int,Int,All,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,GtInt,Int,LtInt,LtInt|All,LtInt,Int,Int,GtInt,LtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,GtInt,All,Int,Int,Int,Int,Int,GtInt,Int|Int,GtInt,Int,LtInt,Int,All,Int,Int,GtInt,All,LtInt,GtInt,Int,GtInt,LtInt,GtInt,Int,Int,LtInt,Int,Int,All,Int,Int,GtInt|GtInt,Int,Int,All,Int,All,Int,GtInt,Int,LtInt,Int,All,LtInt,LtInt,GtInt,GtInt,Int,All,Int,LtInt,Int,Int,Int,All,LtInt|GtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,All,Int,All,Int,Int,LtInt,LtInt,LtInt,Int,Int,Int,GtInt,GtInt,Int,Int|]; 
constraint smart_table([x23,x11,x21,x13,x28,x10,x18,x9,x17,x6,x36,x31,x38,x0,x27,x26,x24,x22,x15,x37,x8,x3,x16,x7,x29],t0,signs0)::uniud;
solve satisfy;