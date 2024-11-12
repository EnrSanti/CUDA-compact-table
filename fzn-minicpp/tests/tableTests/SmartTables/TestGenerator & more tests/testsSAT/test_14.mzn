%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 70..177 : x0;
var 26..181 : x1;
var 81..222 : x2;
var 46..152 : x3;
var 65..100 : x4;
var 37..202 : x5;
var 35..188 : x6;
var 48..181 : x7;
var 41..166 : x8;
var 25..175 : x9;
var 82..129 : x10;
var 67..176 : x11;
var 56..249 : x12;
var 55..72 : x13;
var 89..137 : x14;
var 67..303 : x15;
var 34..156 : x16;
var 64..262 : x17;
var 1..110 : x18;
var 59..166 : x19;
var 60..102 : x20;
var 11..85 : x21;
var 75..171 : x22;
var 66..102 : x23;
var 49..264 : x24;
var 26..140 : x25;
var 26..78 : x26;
var 10..117 : x27;
var 87..268 : x28;
var 13..243 : x29;
var 89..111 : x30;
var 71..149 : x31;
var 88..115 : x32;
var 70..279 : x33;
var 88..267 : x34;
var 11..255 : x35;
array [int,int] of 1..303 : t0=[|94,93,65,46,249,125,159,85,181,123,33,110,146,108,103,84,109,98,52,114,158,103,50|
89,91,173,83,295,66,146,88,225,137,51,253,115,106,89,133,78,147,81,123,95,5,54|
96,102,127,135,296,77,160,100,65,141,41,188,133,114,104,158,112,198,64,116,150,95,15|
113,140,159,201,186,115,151,67,99,110,86,90,151,105,174,111,97,233,70,92,145,95,58|
108,103,135,160,67,59,26,73,157,107,18,114,139,108,94,150,162,79,96,115,62,76,27|
99,257,58,120,271,111,117,92,64,76,64,88,97,111,154,130,140,95,138,108,133,71,66|
129,169,92,185,175,144,93,89,178,88,90,247,92,112,157,102,72,180,46,120,53,10,20|
85,173,201,83,151,94,112,68,188,112,28,106,143,88,124,111,107,178,98,106,90,11,25|
124,156,69,92,80,80,150,73,235,114,64,268,84,97,83,101,162,79,42,116,50,28,23|
128,182,87,208,167,60,101,99,78,82,66,101,107,96,68,162,120,227,116,100,99,84,83|
111,103,63,104,286,113,67,67,99,77,75,234,83,115,175,157,176,236,136,118,27,13,60|
83,258,50,40,240,154,92,75,229,74,22,205,159,108,68,139,83,204,86,92,114,30,23|
93,111,45,243,70,108,147,98,68,147,101,123,118,99,140,164,122,197,147,96,152,40,13|
103,135,64,20,182,123,181,79,172,115,56,203,108,95,161,111,168,52,96,105,53,64,50|
126,195,45,89,243,54,122,73,81,116,102,235,82,106,96,70,177,155,75,98,46,4,30|
127,216,74,178,170,160,63,91,129,109,18,199,146,99,89,140,109,117,124,101,49,47,52|
128,140,184,243,175,160,177,68,210,90,83,198,94,90,115,69,133,134,142,116,175,88,34|
104,120,198,79,197,141,118,91,177,99,63,104,151,110,75,104,150,213,64,132,56,3,14|
124,249,111,109,229,172,60,81,217,102,93,164,132,112,74,109,158,64,64,132,120,29,62|
82,129,194,66,75,155,113,75,115,82,16,199,95,115,114,42,96,218,51,108,129,110,50|
129,157,154,174,141,152,107,96,193,78,49,216,169,110,98,76,133,187,91,101,34,12,71|
98,181,58,54,298,159,66,79,211,96,69,205,106,112,78,131,80,197,126,132,92,51,56|
95,128,99,198,174,71,168,79,256,103,36,102,148,90,157,91,90,206,121,94,129,69,69|
93,168,150,71,204,124,180,69,65,123,99,185,75,96,83,136,76,256,97,100,57,98,54|
124,251,162,77,150,71,76,89,156,94,29,135,161,110,121,132,70,177,77,100,39,78,29|
112,135,58,193,211,55,164,66,95,71,72,144,140,114,128,99,175,206,118,123,80,79,62|
110,266,132,68,156,57,29,89,203,95,81,152,128,88,166,149,152,183,150,111,36,108,73|
119,146,110,110,280,117,107,94,123,86,78,113,91,96,140,44,150,78,51,91,106,59,13|
82,149,61,68,293,135,151,97,194,129,26,111,94,97,123,64,143,166,147,95,161,14,17|
108,148,149,86,115,50,30,90,254,114,109,162,98,108,160,48,90,184,79,125,101,43,65|
87,181,124,215,84,60,69,85,252,90,109,265,122,111,132,155,176,247,95,89,96,53,31|
123,208,117,56,75,140,99,85,242,99,59,176,149,99,77,95,105,130,59,119,44,84,55|
128,236,189,162,130,64,158,81,218,101,49,105,141,99,133,92,122,79,132,130,103,108,62|
91,253,46,196,265,173,128,69,166,127,14,115,86,108,116,140,73,91,97,108,66,98,39|
116,157,49,240,190,93,90,84,197,96,107,119,139,88,88,150,142,95,77,120,123,46,85|
82,191,85,197,175,101,157,100,84,114,110,164,140,98,89,68,105,97,105,96,120,51,63|
112,146,177,149,286,176,153,69,135,108,116,172,129,108,132,119,127,77,130,129,169,44,43|
96,176,75,16,227,181,76,67,190,107,74,267,112,91,88,148,140,159,45,90,122,15,59|
114,168,201,233,85,107,107,99,226,132,57,92,166,105,162,156,161,60,139,94,140,96,52|
108,171,67,223,114,176,52,73,78,126,106,261,166,111,86,130,122,139,121,135,120,109,71|
100,153,63,56,92,121,154,74,190,123,50,111,83,90,73,115,88,222,101,94,143,90,49|
129,247,180,97,155,105,137,93,243,115,33,209,142,99,116,44,86,257,125,109,43,28,43|
89,250,117,99,166,163,70,78,159,74,103,87,160,88,111,89,136,250,144,103,150,22,49|
100,196,147,155,129,130,169,77,177,127,47,149,90,95,144,143,88,120,82,98,48,16,49|
121,132,184,237,98,180,116,90,156,101,46,116,131,105,142,158,175,116,100,93,137,66,34|
87,259,95,53,89,70,108,100,171,147,76,92,160,93,127,129,132,62,131,96,87,1,83|
114,115,189,113,216,124,150,91,186,128,107,218,127,93,75,139,166,66,53,107,154,9,80|
113,217,104,66,98,153,75,95,248,75,36,115,156,94,170,166,88,162,83,92,28,107,61|
89,122,115,115,83,65,64,68,205,85,52,181,97,113,81,54,158,53,96,121,42,46,24|
118,183,119,197,231,135,96,75,255,134,97,194,124,104,131,107,124,191,102,125,137,102,72|
105,97,159,61,273,75,44,73,198,138,31,102,115,110,78,66,176,53,139,119,139,61,20|
103,167,185,207,167,173,64,65,210,82,56,188,149,93,85,55,125,189,42,115,51,99,38|
107,255,102,118,286,163,122,87,140,101,32,128,156,103,128,89,87,246,84,101,165,97,78|
100,244,68,240,225,117,84,68,242,80,50,204,75,110,167,67,133,185,45,131,42,34,15|
115,190,147,143,238,72,53,76,260,139,67,193,111,115,119,46,154,56,80,125,137,72,83|
111,168,117,105,154,55,68,89,207,148,25,176,127,115,89,62,98,62,47,135,67,67,23|
82,126,155,15,172,175,74,97,84,136,91,95,152,106,135,60,151,192,152,104,140,32,17|
107,196,139,123,288,179,50,90,208,78,94,168,157,111,168,131,75,101,60,117,74,39,71|
101,181,162,127,279,86,56,92,98,78,105,192,114,114,169,94,147,141,137,113,40,58,81|
87,264,53,61,183,148,90,75,150,128,101,172,162,101,131,79,73,253,142,121,30,91,43|
105,172,41,216,297,59,134,94,94,89,22,238,159,91,90,125,165,251,131,116,79,32,42|
88,184,133,47,303,162,132,95,218,104,91,132,98,111,142,80,125,132,90,121,93,12,21|
107,240,171,26,164,141,33,79,111,101,90,254,108,109,147,83,91,107,104,111,111,4,34|
103,171,147,119,120,153,82,68,91,124,116,181,124,98,108,143,168,110,94,137,42,5,43|
127,232,166,38,97,111,41,83,160,122,68,209,80,92,121,118,88,120,138,137,82,27,73|
126,232,174,123,240,170,161,95,108,120,69,210,151,112,78,106,91,91,129,116,29,83,20|
128,131,88,129,201,139,170,96,232,78,33,88,170,105,172,134,117,222,101,98,136,82,74|
86,174,166,32,241,111,156,85,173,84,48,104,141,103,89,166,125,257,50,97,50,55,64|
107,156,106,46,168,142,102,93,210,115,113,188,157,94,116,79,75,194,47,117,36,10,68|
93,133,101,107,297,52,163,88,122,145,76,188,88,89,90,55,149,73,95,90,41,45,15|
97,95,90,106,264,87,76,91,68,71,23,215,141,90,87,163,134,107,40,130,169,18,70|
97,204,135,222,116,163,93,83,117,120,14,106,121,110,90,82,141,76,79,137,117,11,73|
111,213,141,162,282,121,98,85,91,103,70,180,122,115,75,147,176,198,141,121,54,84,63|
129,257,79,141,106,110,106,80,228,116,109,201,83,106,140,69,98,209,122,104,169,16,12|
88,194,93,34,296,136,138,78,159,127,104,192,105,109,149,111,174,244,115,99,96,65,66|
98,231,179,73,67,53,107,65,150,102,36,144,158,89,167,93,125,167,88,132,31,53,41|
87,232,164,243,266,168,105,79,230,100,24,202,136,90,132,88,135,70,123,101,81,85,80|
101,195,57,229,194,55,132,80,68,114,66,172,105,102,117,113,93,123,95,137,89,72,34|
129,234,87,39,137,103,83,84,119,88,49,98,100,104,130,74,177,178,153,105,118,15,24|
118,156,149,101,240,137,163,67,107,140,76,183,136,102,86,77,114,79,99,117,56,63,39|
126,100,52,130,217,100,125,92,207,117,89,166,122,89,98,80,71,223,67,109,109,3,34|
115,109,184,144,87,82,35,74,191,79,78,181,98,109,87,105,167,72,139,119,102,12,83|
120,170,202,105,228,105,86,65,234,96,115,216,92,102,158,159,144,207,135,96,133,106,52|
126,200,114,195,87,141,154,89,256,120,43,170,129,107,129,83,102,212,79,127,53,105,75|
85,187,157,200,213,156,98,89,235,75,112,108,77,111,100,51,143,125,96,109,56,37,54|
127,98,164,111,85,177,82,83,152,138,67,87,128,92,95,60,96,240,140,100,29,66,20|
123,99,165,202,140,51,63,75,204,71,22,221,122,95,152,46,91,133,121,120,62,66,20|
97,139,55,213,169,147,61,83,70,124,84,234,89,93,107,62,85,64,124,110,39,1,73|
93,132,39,34,223,178,135,79,257,90,25,204,117,109,104,148,142,221,89,135,85,21,20|
128,226,123,94,289,73,116,97,151,105,99,201,168,102,67,41,91,191,110,107,25,49,82|
116,163,87,102,232,74,105,78,71,149,10,108,107,107,175,71,158,56,142,110,131,65,68|
125,225,152,87,91,154,130,91,258,97,55,263,160,106,127,75,172,88,100,130,99,87,50|]; 
array [int,int] of SmartTableOp: signs0=[|All,Int,LtInt,LtInt,All,Int,Int,Int,All,Int,Int,All,Int,Int,Int,All,Int,Int,Int,GtInt,GtInt,Int,Int|Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int,LtInt,GtInt,LtInt,Int,Int,Int,Int,Int,All,Int,LtInt,LtInt,Int,All|Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,Int,LtInt,GtInt,All,Int,LtInt,GtInt,Int,Int,Int,GtInt,Int,Int|Int,GtInt,GtInt,GtInt,All,Int,Int,Int,Int,Int,All,Int,All,Int,LtInt,All,Int,All,Int,Int,Int,Int,Int|Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,All,GtInt,LtInt,Int,Int,LtInt,All,Int,Int,Int|Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,Int,Int,LtInt,Int,Int,Int,LtInt,All,All,Int,Int,All|Int,Int,Int,Int,All,LtInt,All,LtInt,Int,Int,GtInt,LtInt,Int,LtInt,Int,All,Int,GtInt,Int,LtInt,Int,Int,Int|GtInt,Int,Int,All,Int,Int,Int,Int,Int,GtInt,LtInt,Int,Int,GtInt,Int,Int,GtInt,All,All,Int,Int,GtInt,All|All,GtInt,Int,GtInt,Int,LtInt,Int,Int,LtInt,Int,Int,Int,Int,LtInt,LtInt,Int,All,Int,Int,GtInt,Int,GtInt,All|Int,All,LtInt,Int,Int,Int,Int,Int,Int,All,LtInt,GtInt,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,Int|Int,Int,Int,Int,All,LtInt,Int,Int,Int,Int,All,LtInt,GtInt,Int,Int,LtInt,GtInt,GtInt,Int,Int,Int,Int,LtInt|Int,Int,All,GtInt,LtInt,LtInt,GtInt,Int,Int,LtInt,Int,Int,LtInt,All,Int,GtInt,LtInt,Int,GtInt,Int,Int,Int,Int|Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,Int,GtInt,LtInt,Int,Int,Int,All,GtInt,LtInt,Int,Int,LtInt,Int|Int,Int,Int,Int,LtInt,LtInt,All,Int,LtInt,Int,Int,LtInt,LtInt,Int,LtInt,Int,LtInt,LtInt,Int,All,LtInt,LtInt,LtInt|Int,Int,Int,Int,Int,LtInt,Int,LtInt,GtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt|Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int|Int,Int,Int,All,Int,GtInt,Int,Int,Int,LtInt,Int,GtInt,Int,Int,Int,LtInt,Int,Int,LtInt,Int,All,GtInt,GtInt|All,Int,Int,Int,Int,GtInt,All,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,GtInt,Int,LtInt,LtInt,Int,LtInt|All,All,Int,Int,Int,All,LtInt,Int,LtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,Int|GtInt,LtInt,All,LtInt,Int,GtInt,GtInt,Int,Int,GtInt,LtInt,GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,All|GtInt,Int,Int,Int,GtInt,Int,All,Int,Int,Int,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,Int,Int,LtInt,Int,GtInt,Int,Int,LtInt,LtInt,Int,GtInt,Int,All,Int,Int,Int,LtInt,Int,LtInt,Int,Int,Int|GtInt,GtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int,Int|GtInt,Int,Int,LtInt,Int,Int,Int,All,Int,All,Int,Int,Int,All,LtInt,LtInt,LtInt,Int,Int,Int,LtInt,GtInt,LtInt|Int,GtInt,Int,Int,All,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,All,Int,Int,Int,LtInt,Int,Int,Int,LtInt|Int,Int,Int,Int,Int,Int,Int,LtInt,All,LtInt,All,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,Int|Int,Int,Int,LtInt,Int,Int,Int,All,GtInt,Int,LtInt,All,All,Int,GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int|Int,GtInt,All,Int,Int,Int,Int,Int,LtInt,LtInt,LtInt,Int,All,Int,All,GtInt,All,LtInt,Int,Int,LtInt,GtInt,Int|LtInt,All,Int,Int,All,Int,Int,Int,LtInt,GtInt,Int,Int,Int,LtInt,Int,All,Int,Int,Int,Int,Int,All,Int|GtInt,GtInt,Int,GtInt,GtInt,Int,Int,LtInt,LtInt,Int,Int,Int,All,Int,LtInt,Int,Int,Int,Int,Int,GtInt,Int,Int|Int,Int,LtInt,All,Int,GtInt,All,All,Int,LtInt,LtInt,Int,Int,All,Int,Int,GtInt,Int,LtInt,All,All,Int,GtInt|Int,Int,Int,All,GtInt,Int,Int,LtInt,Int,Int,GtInt,GtInt,LtInt,Int,Int,Int,LtInt,Int,LtInt,LtInt,LtInt,GtInt,LtInt|Int,Int,GtInt,Int,LtInt,LtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,LtInt,LtInt,Int,LtInt|LtInt,Int,Int,Int,Int,Int,All,LtInt,LtInt,LtInt,LtInt,Int,All,Int,Int,Int,GtInt,Int,Int,GtInt,Int,LtInt,Int|Int,Int,Int,Int,Int,Int,Int,GtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,Int|All,Int,Int,GtInt,GtInt,Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,All,LtInt,GtInt,Int,Int,GtInt,Int|Int,Int,All,LtInt,LtInt,All,Int,Int,Int,LtInt,Int,Int,GtInt,GtInt,GtInt,LtInt,Int,Int,Int,Int,Int,LtInt,Int|Int,GtInt,Int,All,Int,GtInt,LtInt,LtInt,GtInt,LtInt,GtInt,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,Int,All,LtInt|Int,Int,GtInt,LtInt,LtInt,LtInt,Int,Int,Int,GtInt,LtInt,Int,GtInt,GtInt,LtInt,Int,LtInt,Int,Int,All,LtInt,Int,Int|Int,Int,GtInt,GtInt,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,LtInt,LtInt,LtInt,Int|Int,All,Int,Int,Int,All,Int,Int,Int,Int,LtInt,All,Int,All,All,Int,GtInt,GtInt,All,Int,GtInt,All,GtInt|Int,Int,Int,LtInt,Int,LtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,LtInt,Int,LtInt,Int,LtInt|LtInt,GtInt,Int,Int,All,LtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,All,LtInt,LtInt,Int,Int,Int|LtInt,Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,Int,GtInt,LtInt,Int,LtInt,LtInt,Int,Int,Int,LtInt,LtInt,All,All|GtInt,LtInt,Int,Int,Int,Int,All,Int,GtInt,Int,Int,All,GtInt,Int,Int,Int,LtInt,Int,GtInt,LtInt,GtInt,GtInt,Int|All,Int,Int,Int,All,All,Int,LtInt,LtInt,Int,Int,GtInt,LtInt,LtInt,All,Int,Int,Int,Int,All,Int,GtInt,Int|Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,Int,GtInt,GtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int,GtInt,GtInt,Int|Int,Int,All,LtInt,GtInt,Int,Int,All,Int,LtInt,All,LtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,GtInt,Int,LtInt|Int,LtInt,LtInt,Int,Int,All,Int,LtInt,Int,Int,GtInt,GtInt,Int,LtInt,Int,Int,LtInt,LtInt,Int,All,Int,Int,Int|GtInt,Int,Int,Int,Int,Int,Int,All,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,All,Int|Int,All,Int,GtInt,GtInt,LtInt,GtInt,Int,Int,Int,All,Int,Int,GtInt,All,Int,Int,Int,Int,LtInt,Int,LtInt,Int|Int,Int,Int,GtInt,Int,Int,All,Int,LtInt,Int,LtInt,GtInt,Int,LtInt,Int,All,Int,Int,Int,GtInt,LtInt,Int,Int|GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int,GtInt,LtInt,Int,Int,Int,Int,All,GtInt,Int,Int|LtInt,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,All,GtInt,Int,All,Int,Int,Int,LtInt|Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,All,Int,Int,Int,GtInt,GtInt,Int,Int,LtInt,Int,GtInt|GtInt,Int,Int,Int,Int,GtInt,All,Int,Int,LtInt,All,Int,All,Int,Int,GtInt,Int,All,Int,Int,Int,LtInt,Int|Int,Int,Int,All,GtInt,Int,Int,LtInt,GtInt,Int,LtInt,All,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,GtInt|Int,LtInt,GtInt,Int,Int,LtInt,Int,Int,Int,All,Int,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int|Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,All,LtInt,Int,LtInt,Int,Int,Int,GtInt,All,LtInt|Int,Int,Int,All,Int,Int,GtInt,All,Int,Int,Int,Int,Int,Int,GtInt,GtInt,Int,Int,GtInt,Int,GtInt,Int,Int|Int,Int,LtInt,Int,Int,Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,Int,LtInt,GtInt,Int,Int,Int,All,GtInt,Int,All|Int,Int,Int,Int,GtInt,Int,Int,All,Int,LtInt,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int|LtInt,GtInt,Int,All,Int,Int,All,Int,Int,All,LtInt,All,Int,All,Int,Int,All,LtInt,Int,GtInt,Int,LtInt,Int|LtInt,Int,Int,Int,LtInt,All,Int,Int,All,LtInt,Int,GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int|All,Int,Int,Int,Int,Int,Int,All,LtInt,Int,Int,Int,GtInt,GtInt,Int,Int,GtInt,Int,Int,Int,LtInt,All,GtInt|GtInt,Int,All,Int,Int,LtInt,GtInt,Int,Int,LtInt,All,Int,LtInt,LtInt,Int,Int,Int,GtInt,Int,GtInt,Int,Int,All|GtInt,Int,LtInt,LtInt,GtInt,LtInt,LtInt,LtInt,GtInt,LtInt,Int,All,Int,Int,Int,All,Int,LtInt,Int,Int,LtInt,GtInt,Int|All,LtInt,Int,Int,All,Int,GtInt,Int,All,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int,LtInt,LtInt,Int,Int,Int|LtInt,All,Int,GtInt,All,Int,Int,GtInt,Int,GtInt,Int,Int,LtInt,Int,LtInt,GtInt,Int,Int,Int,Int,LtInt,GtInt,Int|Int,Int,Int,LtInt,GtInt,LtInt,Int,Int,LtInt,GtInt,Int,All,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt|Int,Int,LtInt,LtInt,GtInt,LtInt,GtInt,GtInt,LtInt,Int,All,Int,Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,Int,GtInt,Int|LtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,All,GtInt,Int,All,All,Int,GtInt,Int,Int,GtInt,GtInt,Int,LtInt,GtInt|Int,GtInt,Int,GtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,All,Int,LtInt,All,Int,All,Int,Int,GtInt,Int|All,Int,GtInt,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,All,Int,Int,All,All|Int,Int,Int,Int,GtInt,GtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,All,GtInt,All,Int,GtInt,GtInt,All|All,Int,LtInt,GtInt,Int,Int,Int,Int,Int,GtInt,All,All,LtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,LtInt,All|Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,GtInt,Int,Int|GtInt,Int,Int,LtInt,Int,GtInt,LtInt,GtInt,Int,Int,Int,LtInt,Int,All,LtInt,Int,LtInt,Int,LtInt,LtInt,Int,All,GtInt|Int,LtInt,Int,LtInt,Int,Int,All,Int,LtInt,GtInt,All,Int,Int,Int,Int,GtInt,Int,Int,GtInt,GtInt,Int,Int,Int|GtInt,Int,Int,GtInt,GtInt,LtInt,Int,GtInt,Int,GtInt,Int,All,Int,Int,GtInt,GtInt,LtInt,Int,Int,Int,Int,All,Int|Int,Int,GtInt,GtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,GtInt,Int,GtInt,Int,LtInt,Int|Int,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,All,LtInt,Int|LtInt,GtInt,GtInt,GtInt,LtInt,LtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,All,GtInt,Int|Int,Int,Int,Int,Int,All,LtInt,All,LtInt,Int,Int,All,Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,GtInt|GtInt,LtInt,Int,All,GtInt,Int,Int,Int,Int,LtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,Int,Int|LtInt,Int,Int,Int,All,GtInt,Int,Int,All,All,GtInt,Int,Int,GtInt,Int,Int,All,GtInt,GtInt,Int,LtInt,Int,Int|Int,Int,Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,LtInt,GtInt,Int,Int|Int,Int,Int,LtInt,LtInt,Int,Int,Int,LtInt,Int,GtInt,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,LtInt,All,GtInt|Int,Int,Int,Int,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,Int,Int,LtInt,Int,GtInt|Int,LtInt,GtInt,LtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,LtInt|All,Int,Int,LtInt,LtInt,GtInt,Int,All,GtInt,All,GtInt,Int,Int,Int,LtInt,Int,Int,Int,LtInt,Int,Int,Int,Int|GtInt,Int,Int,LtInt,Int,All,Int,Int,GtInt,Int,LtInt,Int,Int,Int,All,GtInt,LtInt,Int,All,Int,Int,GtInt,GtInt|]; 
constraint smart_table([x10,x34,x5,x29,x15,x7,x1,x4,x17,x31,x27,x28,x22,x32,x11,x8,x0,x24,x16,x14,x9,x18,x21],t0,signs0)::uniud;
constraint x10<112;
constraint not(x34=127);
constraint x5<199;
constraint not(x29=68);
constraint not(x15=150);
constraint not(x7=58);
constraint x1<97;
constraint x4>90;
constraint not(x17=94);
constraint not(x31=139);
constraint not(x27=96);
constraint not(x28=210);
constraint x22=113;
constraint not(x32=93);
constraint not(x11=120);
constraint x8>115;
solve satisfy;