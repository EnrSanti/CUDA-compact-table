%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 36..249 : x0;
var 57..227 : x1;
var 58..204 : x2;
var 72..176 : x3;
var 39..222 : x4;
var 79..318 : x5;
var 27..161 : x6;
var 87..318 : x7;
var 41..100 : x8;
var 26..230 : x9;
var 67..212 : x10;
var 90..183 : x11;
var 22..51 : x12;
var 90..283 : x13;
var 19..133 : x14;
var 43..281 : x15;
var 47..281 : x16;
var 68..185 : x17;
var 56..213 : x18;
var 30..253 : x19;
var 69..92 : x20;
var 20..113 : x21;
var 38..196 : x22;
var 62..175 : x23;
var 79..305 : x24;
var 38..191 : x25;
var 70..94 : x26;
var 69..210 : x27;
var 56..271 : x28;
var 46..109 : x29;
var 2..240 : x30;
var 7..247 : x31;
var 51..179 : x32;
var 25..71 : x33;
var 82..313 : x34;
var 42..84 : x35;
var 64..223 : x36;
var 32..50 : x37;
var 72..151 : x38;
var 9..163 : x39;
var 7..185 : x40;
var 82..273 : x41;
var 52..290 : x42;
var 53..129 : x43;
var 52..231 : x44;
var 81..98 : x45;
var 53..152 : x46;
var 88..269 : x47;
var 48..185 : x48;
var 69..292 : x49;
var 4..205 : x50;
var 64..188 : x51;
var 20..91 : x52;
array [int,int] of 2..318 : t0=[|161,104,257,141,106,119,68,45,52,62,286,168,313,202,150,87,98,172,87,174,158,63,58,190,115|
145,151,139,89,67,137,211,32,58,53,246,168,257,162,59,146,72,96,96,148,150,63,107,99,128|
59,131,267,163,25,129,209,67,85,86,91,145,186,91,127,103,34,149,54,113,99,57,164,233,46|
179,152,104,67,107,157,177,63,46,116,131,96,197,166,224,85,59,78,116,189,77,72,132,117,164|
146,251,237,121,72,181,136,31,52,80,150,103,305,69,91,79,55,140,84,64,140,105,135,138,133|
201,231,57,166,84,142,133,35,89,61,128,122,300,228,154,102,61,111,56,117,126,47,58,63,92|
102,171,113,79,26,137,220,30,22,73,248,94,302,125,107,126,31,100,71,105,55,51,163,42,19|
251,141,129,141,100,69,40,33,83,94,275,116,184,48,199,115,60,75,107,70,168,61,49,141,68|
273,192,103,100,81,164,192,45,48,96,117,76,307,179,86,79,85,163,84,58,61,106,117,24,161|
105,254,82,167,47,177,244,30,54,103,283,102,243,127,89,116,117,153,121,51,104,101,133,75,14|
133,180,189,113,63,160,106,51,74,81,268,81,283,127,205,150,125,116,53,39,87,51,149,231,17|
151,260,202,168,99,151,60,33,58,138,161,170,91,253,215,120,77,150,78,134,131,78,190,219,136|
147,256,151,170,38,144,89,64,86,111,275,165,201,60,222,98,102,121,90,168,119,71,136,30,108|
236,257,167,183,101,88,218,41,58,131,286,80,225,71,60,147,102,154,82,136,150,76,99,101,89|
134,267,78,204,105,73,159,51,73,142,150,61,166,122,207,100,95,82,61,157,67,65,100,203,56|
180,161,156,154,103,129,69,48,40,98,304,99,313,116,225,142,42,162,56,94,123,104,99,194,142|
257,200,133,207,25,98,111,37,84,53,209,145,227,195,89,114,19,158,107,122,67,105,76,217,155|
115,213,79,71,63,137,101,36,64,123,80,69,237,159,72,80,22,165,90,15,118,56,57,63,181|
77,263,94,158,107,106,223,64,65,126,108,107,184,72,161,97,82,101,111,70,72,64,146,104,86|
261,251,117,136,80,152,71,62,62,115,150,164,190,131,71,148,60,74,91,150,54,49,97,57,75|
92,151,211,157,32,130,113,36,66,104,247,173,116,70,188,126,121,98,127,139,95,50,169,85,146|
72,89,267,82,108,120,165,50,90,77,207,94,286,40,99,83,113,162,85,14,156,91,136,49,172|
259,213,72,96,79,114,80,50,77,142,196,142,183,147,142,148,73,147,59,65,120,94,70,79,146|
246,255,194,155,95,78,65,28,59,150,163,98,208,146,176,150,103,152,60,85,157,53,153,217,57|
275,136,260,184,24,179,220,40,58,111,187,169,110,164,113,111,41,151,64,104,138,88,57,116,70|
62,98,173,157,112,176,197,52,44,107,87,112,259,78,158,124,76,129,84,114,99,108,73,119,43|
153,243,56,177,66,151,119,35,66,106,232,92,191,105,222,99,82,102,78,35,81,95,163,202,162|
235,269,193,206,79,142,49,30,59,67,218,191,310,89,104,120,97,114,78,137,140,66,43,79,108|
149,138,169,198,46,136,123,55,86,102,149,98,245,123,122,73,62,73,93,95,145,104,80,124,79|
147,159,268,178,35,153,202,67,36,121,287,199,128,214,149,149,20,173,90,145,60,49,110,116,20|
141,270,123,219,62,88,188,37,60,61,137,100,165,171,145,74,24,158,100,33,95,66,132,91,93|
281,106,167,141,78,133,177,33,88,72,113,177,91,59,133,128,56,125,68,84,89,68,82,47,62|
183,246,253,152,65,95,193,36,21,148,101,66,170,40,216,141,67,145,61,201,71,82,130,24,41|
197,175,79,87,93,68,209,55,20,96,90,126,182,194,99,80,98,100,102,127,83,108,87,180,121|
192,271,225,111,108,144,149,57,65,143,186,180,177,219,86,75,114,125,58,169,159,50,119,159,60|
133,181,212,118,60,73,91,36,40,58,190,64,304,68,150,134,112,94,99,71,94,99,156,149,162|
274,88,118,136,53,156,248,35,81,113,283,60,255,117,136,85,107,98,67,115,58,65,53,3,172|
246,125,100,108,36,169,235,33,59,108,168,107,187,107,129,140,126,111,61,113,110,101,153,221,40|
268,179,206,74,108,91,173,64,79,81,153,129,166,60,87,126,88,92,95,112,172,71,182,84,93|
181,252,263,155,84,175,46,55,35,99,154,146,101,107,217,117,114,115,96,11,110,78,103,18,152|
222,249,263,174,61,94,204,46,38,62,281,144,182,65,166,148,130,101,107,24,86,70,98,63,61|
110,181,218,75,84,127,201,69,24,112,233,95,270,249,220,137,102,107,88,114,50,52,44,193,66|
138,254,100,208,110,128,229,66,91,74,133,173,90,243,133,130,65,141,73,174,171,106,149,108,35|
206,105,213,128,40,117,50,51,84,95,156,138,132,110,166,138,64,98,80,65,85,62,70,84,14|
258,221,256,169,89,133,223,69,72,120,239,170,130,37,118,126,95,170,117,203,120,58,48,94,134|
106,99,82,162,77,125,213,53,24,109,167,179,275,96,186,115,124,75,97,149,141,104,126,47,110|
251,194,241,141,47,71,191,52,27,87,153,114,187,232,119,109,101,161,70,182,138,102,175,195,23|
163,200,140,152,62,176,79,68,65,80,157,76,249,237,173,94,73,80,113,28,113,89,136,109,172|
151,210,111,139,91,174,239,46,62,90,297,64,245,144,175,91,58,127,121,84,82,60,68,213,155|
132,123,168,138,77,69,68,49,82,54,300,200,185,230,121,144,89,87,127,154,57,48,187,122,59|
84,134,105,158,69,135,210,33,72,131,162,196,174,206,67,88,73,134,117,174,115,55,83,181,92|
72,157,153,121,105,102,102,45,69,94,102,74,218,43,102,102,43,125,88,186,105,98,165,134,178|
155,183,153,158,84,112,162,30,56,73,255,193,298,131,63,150,102,144,65,114,146,99,129,110,65|
155,218,238,131,76,126,118,26,24,79,84,114,283,100,129,135,90,175,121,146,169,70,104,149,164|
253,109,143,75,23,87,48,26,57,68,285,67,220,43,163,79,71,114,102,69,68,74,162,102,89|
101,135,161,139,99,86,189,38,39,129,241,137,230,172,107,94,68,103,119,110,78,106,62,82,79|
272,121,56,159,82,68,82,69,51,113,131,91,269,32,93,144,114,157,124,30,98,84,81,124,144|
58,216,253,139,61,71,133,48,53,88,298,67,154,241,149,77,38,155,78,138,58,103,41,75,86|
212,106,265,177,27,102,213,38,64,85,120,101,305,77,199,138,91,151,64,4,143,104,116,222,119|
169,242,228,149,104,71,222,67,49,102,96,161,100,155,161,85,45,103,91,59,114,78,126,189,22|
66,173,165,121,64,167,74,37,82,126,209,189,200,124,203,130,39,88,113,116,127,83,171,188,178|
204,233,267,208,94,184,240,37,26,135,258,195,132,223,186,98,27,100,64,14,66,75,156,172,38|
267,123,95,80,42,125,234,50,77,65,184,151,169,226,139,101,79,98,128,149,185,89,135,200,9|
279,172,121,194,25,167,48,37,42,134,161,92,134,163,196,130,130,100,112,40,77,55,179,88,151|
121,111,121,88,78,104,128,30,55,118,277,156,137,161,182,92,99,140,83,140,88,63,163,78,130|
247,99,259,134,24,91,148,32,46,89,170,136,311,222,196,118,128,126,118,91,65,68,128,131,22|
147,207,79,174,52,169,66,49,40,59,106,100,138,104,123,118,96,90,62,99,96,102,170,107,88|
109,144,268,215,29,120,76,25,63,94,132,162,156,228,202,82,110,81,121,184,67,65,113,109,50|
62,144,206,190,39,136,242,32,87,80,122,150,124,177,152,132,110,110,76,195,83,68,80,75,29|
210,147,139,216,99,107,181,65,51,133,107,108,215,229,117,116,22,111,69,54,62,66,152,97,181|
114,197,262,211,108,76,70,53,83,57,128,62,281,77,160,131,31,158,70,47,93,105,60,88,35|
138,152,213,143,91,176,207,32,32,138,84,115,116,82,88,150,29,145,103,134,97,105,143,42,83|
233,97,263,123,84,106,185,41,24,120,260,198,314,70,118,146,67,160,64,7,90,60,156,190,79|
265,133,230,101,60,182,192,25,47,79,133,158,241,222,139,122,133,167,129,179,59,107,62,2,164|
222,97,156,141,76,165,146,28,82,65,265,176,97,164,192,82,36,169,78,146,50,58,144,107,94|
134,134,227,81,38,122,179,39,82,55,197,141,288,215,118,115,114,80,54,199,94,58,124,48,162|
133,158,210,157,24,111,170,67,74,104,100,113,148,155,94,136,46,91,105,95,183,67,156,24,135|
160,149,118,176,106,97,240,50,78,77,158,193,311,228,90,112,42,136,108,105,100,50,75,105,57|
273,267,63,216,47,127,175,49,38,72,208,143,277,164,88,149,40,151,127,40,183,97,123,12,27|
252,83,159,171,61,119,45,65,43,137,207,155,230,95,120,130,80,91,88,42,68,107,142,234,88|
115,180,220,198,46,103,142,62,31,95,216,87,193,81,68,140,68,94,53,135,175,95,156,227,138|
259,135,80,105,92,183,44,28,33,78,217,82,278,209,189,82,38,142,126,142,90,65,176,33,55|
161,229,205,93,63,81,155,57,59,107,88,174,250,194,100,120,119,163,99,75,126,47,133,70,106|
93,119,85,219,35,92,196,54,21,68,143,131,170,59,174,102,50,124,60,135,171,59,183,101,66|
126,194,97,76,58,96,77,63,32,138,210,188,100,240,65,138,132,106,105,84,170,58,88,130,15|
141,108,266,153,74,152,132,46,88,102,124,90,220,144,206,139,121,111,116,37,170,57,159,48,174|
58,243,210,108,96,112,233,28,70,73,154,183,286,70,176,114,88,145,109,167,169,104,115,127,96|
177,135,206,186,49,141,197,61,41,102,216,123,182,137,191,105,106,172,75,134,92,74,90,112,26|
149,101,205,194,74,166,65,52,79,102,284,127,217,49,224,140,40,130,77,11,99,85,116,118,133|
135,175,84,187,59,178,124,38,88,125,122,169,276,149,179,83,121,86,70,15,79,47,86,155,102|
132,220,152,151,104,179,185,30,74,104,210,157,182,150,101,150,46,92,120,153,119,76,78,115,148|
111,170,240,210,42,107,82,39,30,122,294,96,166,101,140,137,38,73,118,100,182,97,181,225,104|
219,254,152,97,50,72,187,66,36,99,123,126,268,95,94,80,112,155,81,34,126,91,132,113,153|
170,135,170,220,47,176,146,26,48,118,193,65,217,96,193,133,106,82,70,7,150,94,158,81,56|
59,98,96,190,108,178,222,64,73,68,172,115,269,253,97,94,112,136,120,166,123,76,50,161,114|
66,218,118,91,106,184,121,35,71,110,282,98,139,73,151,90,133,153,90,81,107,79,84,21,54|
133,254,249,119,98,175,156,34,69,117,166,101,223,50,92,100,110,72,85,75,118,74,116,107,72|
54,246,104,134,31,145,221,47,71,63,293,171,188,53,145,87,19,103,104,34,116,95,104,156,10|
227,218,262,84,39,149,93,26,23,92,253,60,164,101,68,87,76,151,82,149,94,78,177,155,43|]; 
array [int,int] of SmartTableOp: signs0=[|Int,LtInt,Int,Int,Int,Int,LtInt,Int,GtInt,LtInt,All,LtInt,LtInt,Int,GtInt,GtInt,GtInt,Int,Int,All,Int,All,LtInt,Int,Int|Int,GtInt,Int,Int,GtInt,Int,Int,Int,Int,GtInt,LtInt,Int,Int,LtInt,Int,Int,LtInt,LtInt,Int,Int,Int,LtInt,Int,Int,Int|Int,Int,Int,LtInt,Int,Int,Int,LtInt,LtInt,Int,LtInt,GtInt,Int,LtInt,Int,GtInt,Int,Int,All,Int,Int,All,Int,LtInt,LtInt|Int,Int,All,Int,Int,GtInt,Int,All,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,All,LtInt,GtInt|Int,GtInt,Int,LtInt,Int,All,LtInt,Int,Int,Int,Int,GtInt,Int,LtInt,LtInt,LtInt,LtInt,LtInt,LtInt,GtInt,Int,LtInt,Int,GtInt,LtInt|GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,All,GtInt,Int,Int,LtInt,All,Int|Int,Int,GtInt,Int,LtInt,Int,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,GtInt,Int,Int,Int,LtInt|Int,Int,Int,All,Int,Int,Int,LtInt,All,GtInt,Int,Int,Int,All,Int,All,All,GtInt,All,Int,All,Int,Int,Int,Int|Int,All,Int,GtInt,LtInt,GtInt,Int,GtInt,GtInt,LtInt,LtInt,GtInt,All,Int,Int,Int,LtInt,Int,GtInt,Int,Int,Int,Int,Int,Int|GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,GtInt,LtInt,Int,Int,LtInt,Int,Int,All,Int,Int,GtInt,GtInt,GtInt,Int,Int|LtInt,Int,All,GtInt,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,GtInt,Int,Int,GtInt,LtInt,LtInt,Int,GtInt,GtInt,Int,Int|Int,All,Int,Int,Int,Int,Int,LtInt,All,Int,LtInt,GtInt,Int,Int,Int,GtInt,GtInt,All,Int,Int,Int,Int,Int,Int,GtInt|GtInt,LtInt,LtInt,Int,Int,All,Int,Int,Int,GtInt,Int,All,GtInt,LtInt,LtInt,LtInt,Int,GtInt,LtInt,GtInt,Int,Int,All,Int,Int|LtInt,Int,All,Int,All,Int,Int,Int,LtInt,Int,Int,LtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,All,Int,Int|GtInt,Int,Int,All,Int,Int,Int,Int,Int,Int,GtInt,Int,All,Int,Int,Int,Int,GtInt,LtInt,Int,All,Int,Int,LtInt,Int|Int,All,GtInt,GtInt,All,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,LtInt,All,LtInt,GtInt,Int,Int,Int,Int|Int,LtInt,Int,Int,GtInt,GtInt,LtInt,Int,Int,LtInt,Int,Int,Int,All,Int,Int,Int,Int,Int,GtInt,All,Int,Int,Int,All|GtInt,Int,Int,GtInt,LtInt,Int,All,Int,Int,Int,All,Int,Int,Int,LtInt,All,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int|GtInt,Int,GtInt,Int,Int,Int,Int,Int,LtInt,LtInt,All,Int,All,Int,LtInt,LtInt,Int,Int,GtInt,LtInt,Int,GtInt,Int,Int,All|Int,Int,Int,All,GtInt,Int,Int,Int,Int,Int,Int,GtInt,LtInt,All,Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,Int|LtInt,Int,GtInt,Int,GtInt,LtInt,LtInt,All,Int,All,Int,GtInt,Int,Int,Int,Int,LtInt,All,Int,All,Int,GtInt,GtInt,Int,LtInt|Int,LtInt,LtInt,Int,GtInt,GtInt,LtInt,Int,Int,All,Int,Int,GtInt,All,Int,LtInt,Int,Int,Int,All,Int,Int,LtInt,Int,Int|Int,Int,LtInt,LtInt,Int,Int,Int,All,Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,All,Int|LtInt,All,LtInt,Int,LtInt,Int,All,Int,Int,Int,Int,GtInt,All,All,GtInt,LtInt,Int,Int,LtInt,GtInt,GtInt,Int,LtInt,Int,Int|LtInt,GtInt,LtInt,LtInt,Int,Int,Int,Int,Int,GtInt,All,Int,Int,All,GtInt,LtInt,GtInt,Int,All,Int,Int,All,LtInt,All,Int|Int,GtInt,GtInt,All,Int,LtInt,LtInt,GtInt,Int,Int,Int,Int,Int,All,GtInt,LtInt,Int,LtInt,All,LtInt,Int,Int,Int,Int,Int|Int,GtInt,Int,Int,GtInt,GtInt,Int,All,All,Int,Int,Int,GtInt,Int,LtInt,LtInt,Int,Int,GtInt,Int,GtInt,LtInt,All,GtInt,Int|Int,Int,Int,GtInt,GtInt,LtInt,All,Int,LtInt,Int,All,Int,GtInt,Int,Int,Int,All,LtInt,Int,Int,Int,Int,Int,GtInt,Int|Int,Int,Int,All,Int,GtInt,Int,All,Int,All,Int,GtInt,Int,LtInt,GtInt,LtInt,Int,GtInt,Int,Int,Int,GtInt,Int,All,Int|Int,LtInt,GtInt,All,All,LtInt,GtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,LtInt,Int|Int,Int,GtInt,All,Int,GtInt,Int,Int,All,Int,Int,Int,Int,LtInt,Int,LtInt,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int|Int,Int,All,GtInt,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,GtInt,LtInt,GtInt|All,GtInt,GtInt,Int,LtInt,Int,Int,Int,GtInt,Int,Int,All,Int,LtInt,LtInt,LtInt,LtInt,Int,Int,LtInt,Int,All,LtInt,GtInt,Int|Int,Int,LtInt,LtInt,Int,Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,GtInt,All,GtInt,LtInt,Int,GtInt,GtInt,LtInt|Int,GtInt,Int,Int,Int,Int,LtInt,GtInt,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,LtInt,Int,Int,LtInt,LtInt,Int,GtInt,LtInt|All,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,All,GtInt,Int,GtInt,GtInt,Int,Int,All,Int,Int,Int,LtInt,All,GtInt,Int|GtInt,Int,LtInt,Int,All,LtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,All,Int,Int,Int,LtInt,All,LtInt,Int|LtInt,LtInt,Int,Int,GtInt,Int,Int,Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,Int,Int,Int,Int|All,Int,LtInt,Int,LtInt,Int,Int,Int,GtInt,Int,GtInt,All,Int,All,All,All,Int,Int,Int,GtInt,Int,LtInt,GtInt,Int,Int|Int,Int,Int,Int,GtInt,GtInt,LtInt,Int,Int,Int,Int,Int,Int,LtInt,LtInt,GtInt,Int,Int,All,Int,Int,Int,Int,Int,Int|Int,Int,Int,Int,Int,All,Int,All,All,Int,Int,LtInt,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,All,Int,GtInt,All|Int,LtInt,Int,LtInt,Int,Int,All,Int,Int,LtInt,GtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,All,All,Int,GtInt,Int,Int|LtInt,Int,Int,GtInt,Int,All,All,LtInt,Int,GtInt,Int,GtInt,All,All,GtInt,LtInt,GtInt,Int,LtInt,GtInt,Int,Int,Int,GtInt,GtInt|All,Int,LtInt,LtInt,Int,Int,Int,Int,Int,LtInt,GtInt,Int,Int,LtInt,Int,Int,Int,GtInt,GtInt,Int,All,Int,Int,LtInt,All|Int,Int,LtInt,All,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int|LtInt,GtInt,LtInt,Int,Int,Int,LtInt,Int,GtInt,Int,LtInt,Int,GtInt,Int,Int,Int,All,Int,GtInt,Int,LtInt,Int,GtInt,Int,Int|Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,GtInt,GtInt,Int,All,Int,Int,Int,Int,Int,Int,GtInt,Int,Int|Int,GtInt,Int,Int,GtInt,Int,Int,Int,LtInt,GtInt,GtInt,Int,Int,Int,All,Int,Int,LtInt,Int,LtInt,Int,GtInt,Int,GtInt,Int|Int,Int,Int,Int,Int,All,All,Int,All,All,All,All,LtInt,GtInt,GtInt,Int,Int,Int,GtInt,Int,LtInt,LtInt,LtInt,Int,Int|Int,All,GtInt,GtInt,Int,Int,Int,Int,Int,LtInt,GtInt,Int,Int,Int,LtInt,Int,Int,LtInt,All,Int,Int,Int,Int,Int,Int|Int,Int,Int,GtInt,Int,GtInt,GtInt,Int,Int,LtInt,Int,Int,LtInt,GtInt,Int,LtInt,All,Int,Int,Int,Int,Int,GtInt,Int,Int|LtInt,Int,LtInt,All,Int,LtInt,GtInt,Int,GtInt,Int,Int,LtInt,Int,GtInt,LtInt,GtInt,GtInt,Int,Int,Int,Int,LtInt,Int,Int,GtInt|LtInt,GtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int,LtInt,Int,GtInt,LtInt,GtInt,Int,Int,LtInt,Int,All,LtInt,GtInt,Int,GtInt,All|GtInt,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,GtInt,GtInt,Int,Int,Int,Int,All,Int,Int,Int,Int,Int,All,Int,Int|Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,LtInt,Int,GtInt,LtInt,GtInt,All,Int,All,Int,Int,Int,GtInt,Int,LtInt|Int,All,Int,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,GtInt,Int,GtInt,Int,Int,LtInt,Int|Int,Int,All,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int,GtInt,Int,GtInt,All,All,Int,All,GtInt|Int,All,All,All,Int,All,Int,LtInt,All,Int,LtInt,All,GtInt,LtInt,Int,Int,Int,Int,Int,Int,All,GtInt,Int,All,Int|Int,All,Int,LtInt,GtInt,Int,Int,Int,Int,Int,All,All,Int,Int,LtInt,Int,Int,All,All,Int,LtInt,All,LtInt,GtInt,Int|Int,Int,GtInt,Int,GtInt,GtInt,Int,Int,Int,Int,LtInt,Int,LtInt,LtInt,Int,LtInt,Int,All,Int,GtInt,Int,Int,Int,Int,Int|Int,All,GtInt,Int,GtInt,Int,GtInt,All,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,LtInt,Int,LtInt,Int,All,GtInt|Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,LtInt,Int,Int,GtInt,Int,LtInt,Int,Int,All|LtInt,LtInt,Int,GtInt,Int,Int,Int,All,Int,All,Int,Int,Int,GtInt,Int,LtInt,Int,Int,Int,All,Int,LtInt,Int,Int,All|Int,Int,GtInt,LtInt,Int,Int,Int,Int,LtInt,LtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,All,Int,LtInt,Int,All,All,GtInt|LtInt,Int,Int,GtInt,GtInt,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,All,Int,Int,Int,Int,Int,LtInt,Int,LtInt,Int|Int,GtInt,Int,LtInt,Int,Int,Int,Int,LtInt,Int,LtInt,Int,LtInt,All,GtInt,Int,Int,Int,Int,GtInt,GtInt,LtInt,Int,Int,LtInt|GtInt,Int,Int,Int,Int,All,All,All,Int,Int,Int,LtInt,All,GtInt,GtInt,Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int|Int,GtInt,Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,GtInt,Int,All,LtInt,GtInt,Int,Int,Int,GtInt,LtInt|LtInt,All,Int,All,Int,Int,Int,LtInt,Int,Int,Int,LtInt,All,Int,LtInt,Int,GtInt,Int,Int,Int,LtInt,LtInt,Int,Int,GtInt|Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,Int,Int,GtInt,All,GtInt,All,Int,Int,All,Int,Int|Int,Int,Int,GtInt,Int,All,GtInt,Int,GtInt,Int,Int,LtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,All|Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,All,LtInt,Int,LtInt,Int,All,Int,LtInt,GtInt,Int,Int,LtInt,GtInt,Int,Int,Int|All,Int,Int,GtInt,All,Int,Int,Int,Int,All,LtInt,Int,Int,All,Int,All,GtInt,Int,Int,GtInt,Int,Int,Int,Int,LtInt|Int,LtInt,All,Int,Int,Int,All,GtInt,LtInt,Int,LtInt,Int,Int,LtInt,Int,LtInt,All,LtInt,LtInt,Int,Int,All,Int,Int,Int|Int,All,Int,Int,Int,All,GtInt,LtInt,GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int|GtInt,Int,All,GtInt,Int,LtInt,LtInt,LtInt,Int,GtInt,All,GtInt,GtInt,LtInt,GtInt,Int,Int,GtInt,Int,GtInt,GtInt,LtInt,Int,LtInt,All|Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt|GtInt,Int,Int,All,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,All,Int,Int,All,GtInt,GtInt,Int,Int,Int,GtInt|Int,GtInt,GtInt,Int,GtInt,GtInt,LtInt,GtInt,Int,Int,Int,All,Int,Int,LtInt,LtInt,Int,All,Int,Int,GtInt,All,Int,LtInt,Int|Int,GtInt,GtInt,Int,LtInt,Int,Int,Int,Int,All,GtInt,Int,LtInt,Int,GtInt,LtInt,Int,Int,Int,Int,Int,All,Int,Int,Int|Int,GtInt,All,GtInt,LtInt,GtInt,Int,LtInt,GtInt,All,Int,All,GtInt,Int,Int,Int,Int,Int,All,GtInt,Int,Int,LtInt,Int,Int|Int,Int,Int,All,All,Int,Int,All,Int,All,GtInt,LtInt,GtInt,Int,Int,Int,Int,Int,GtInt,GtInt,Int,All,GtInt,Int,Int|All,LtInt,Int,LtInt,Int,Int,LtInt,Int,LtInt,Int,GtInt,Int,Int,LtInt,GtInt,Int,Int,Int,All,Int,Int,Int,Int,All,Int|Int,GtInt,GtInt,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,LtInt,LtInt,All,LtInt,Int,Int|LtInt,Int,GtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,All,LtInt,Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,LtInt|Int,Int,Int,GtInt,Int,GtInt,GtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,All,LtInt,GtInt,Int,Int,LtInt,LtInt,Int|GtInt,Int,Int,Int,Int,LtInt,Int,Int,All,LtInt,Int,Int,Int,GtInt,Int,Int,GtInt,GtInt,All,GtInt,Int,Int,Int,Int,Int|Int,GtInt,GtInt,LtInt,Int,LtInt,Int,GtInt,LtInt,Int,Int,Int,Int,All,Int,LtInt,Int,LtInt,GtInt,GtInt,All,LtInt,Int,LtInt,Int|LtInt,LtInt,Int,GtInt,GtInt,LtInt,All,LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,GtInt,All,Int,Int|Int,LtInt,All,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,All,Int,Int,Int,LtInt,All,LtInt,LtInt,Int,GtInt,Int,Int|All,Int,GtInt,All,GtInt,LtInt,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int,Int,GtInt,Int,LtInt,Int,Int,Int,LtInt,Int,GtInt|LtInt,Int,Int,Int,LtInt,LtInt,Int,GtInt,Int,LtInt,All,LtInt,LtInt,Int,Int,GtInt,LtInt,Int,GtInt,Int,Int,Int,Int,Int,LtInt|GtInt,Int,Int,Int,Int,Int,Int,All,GtInt,LtInt,Int,Int,All,Int,Int,Int,Int,Int,LtInt,Int,Int,LtInt,All,Int,Int|Int,All,Int,GtInt,LtInt,Int,GtInt,Int,All,Int,All,LtInt,GtInt,Int,Int,LtInt,GtInt,Int,Int,LtInt,GtInt,Int,Int,Int,Int|Int,Int,Int,LtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,All,Int,Int,LtInt,Int,Int,All,Int,GtInt,GtInt,Int,Int,Int|GtInt,Int,Int,Int,Int,All,GtInt,GtInt,LtInt,LtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,LtInt,All,LtInt|Int,All,Int,GtInt,Int,GtInt,LtInt,LtInt,All,All,Int,GtInt,Int,GtInt,Int,GtInt,LtInt,GtInt,Int,GtInt,Int,Int,Int,GtInt,LtInt|Int,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,All,Int,Int,Int,LtInt,All,Int,Int,All,GtInt,Int,LtInt|Int,Int,GtInt,Int,All,Int,Int,All,LtInt,GtInt,Int,Int,GtInt,GtInt,Int,LtInt,Int,Int,LtInt,GtInt,All,Int,Int,Int,Int|]; 
constraint smart_table([x16,x41,x28,x36,x21,x17,x0,x33,x52,x46,x24,x2,x7,x19,x1,x38,x14,x3,x43,x50,x48,x29,x25,x30,x40],t0,signs0)::uniud;
constraint not(x16=164);
constraint not(x41=262);
constraint not(x28=158);
constraint not(x36=142);
constraint x21>70;
solve satisfy;