%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 39..143 : x0;
var 23..96 : x1;
var 22..106 : x2;
var 14..146 : x3;
var 70..292 : x4;
var 39..73 : x5;
var 87..199 : x6;
var 86..263 : x7;
var 46..122 : x8;
var 31..100 : x9;
var 19..244 : x10;
var 67..302 : x11;
var 26..64 : x12;
var 18..205 : x13;
var 47..206 : x14;
var 62..202 : x15;
var 47..170 : x16;
var 25..101 : x17;
var 83..95 : x18;
var 51..254 : x19;
var 79..230 : x20;
var 65..135 : x21;
var 19..256 : x22;
var 19..252 : x23;
var 12..245 : x24;
var 19..253 : x25;
var 71..238 : x26;
var 15..228 : x27;
var 21..163 : x28;
var 17..208 : x29;
var 36..174 : x30;
var 20..159 : x31;
var 30..95 : x32;
var 14..83 : x33;
var 69..214 : x34;
var 28..127 : x35;
var 70..80 : x36;
var 66..145 : x37;
var 38..202 : x38;
var 36..108 : x39;
var 88..99 : x40;
var 71..126 : x41;
var 77..288 : x42;
var 80..92 : x43;
var 76..276 : x44;
var 20..59 : x45;
var 43..261 : x46;
var 90..166 : x47;
var 20..83 : x48;
var 77..107 : x49;
var 65..262 : x50;
var 15..42 : x51;
var 89..165 : x52;
var 20..202 : x53;
var 10..170 : x54;
array [int,int] of 10..302 : t0=[|96,85,120,68,143,73,72,113,36,34,184,40,94,288,107,130,89,109,84,186,89,22,15,71,123,161,112,61,198,66,49,90,57,267,242,153,106|
78,132,73,113,165,54,142,79,74,56,171,77,91,219,96,120,87,59,62,112,85,54,73,78,100,121,247,57,135,217,169,124,85,248,216,44,83|
102,120,75,159,135,64,167,59,73,48,118,29,94,96,126,117,91,66,97,121,84,24,35,79,84,155,171,91,148,185,74,128,31,267,163,70,62|
98,105,122,185,135,43,139,152,24,26,71,27,88,285,94,83,81,111,59,150,90,27,41,79,112,118,218,41,196,110,123,133,51,230,160,70,92|
77,149,68,127,198,45,158,256,53,51,124,42,93,282,77,127,83,98,72,172,84,22,96,76,109,119,172,48,181,204,172,92,50,117,260,168,27|
85,105,130,137,155,33,216,179,54,63,140,44,99,281,86,96,82,79,86,109,83,40,40,73,116,93,223,38,119,221,160,134,74,213,92,114,160|
87,81,54,129,133,61,107,136,61,36,116,80,90,179,71,113,89,87,76,89,89,22,122,74,71,128,252,53,121,68,147,69,81,140,197,223,123|
77,85,74,201,190,60,199,36,68,38,64,58,97,179,135,103,89,69,37,127,86,48,111,80,104,102,152,47,151,201,37,76,90,147,133,139,51|
106,42,50,50,110,45,69,63,33,57,104,69,89,89,117,92,80,79,42,145,87,26,64,78,116,94,122,57,187,40,38,141,34,205,92,196,108|
88,66,68,65,200,71,159,38,55,44,104,57,98,280,138,85,90,108,93,161,85,50,102,80,93,148,230,59,115,156,156,62,31,143,147,103,123|
79,92,159,170,115,35,141,103,49,27,74,29,93,203,137,93,85,85,93,113,86,26,24,73,75,120,187,58,170,192,61,76,51,82,84,66,24|
101,24,116,38,151,38,100,62,57,43,60,79,94,99,67,93,89,66,80,77,90,42,51,77,80,161,184,89,126,18,145,59,52,104,260,159,131|
92,159,86,112,101,42,150,114,32,62,46,75,96,92,129,89,86,93,97,81,93,46,41,79,100,108,208,107,142,219,39,50,63,129,235,195,128|
82,72,77,64,171,73,66,195,49,59,157,73,96,274,73,82,82,50,34,113,83,43,138,71,106,111,91,52,144,26,153,113,50,215,213,86,117|
99,27,152,130,111,49,233,248,22,28,96,43,98,113,74,100,82,71,38,119,83,27,138,79,87,150,225,47,125,196,167,58,45,82,127,60,93|
90,95,84,46,193,80,71,226,79,34,127,86,99,263,115,84,90,116,37,107,94,49,130,79,86,140,110,99,185,53,107,70,56,297,271,201,148|
99,87,38,56,111,83,114,177,32,38,51,78,99,292,118,91,80,69,41,93,93,39,72,71,118,120,105,89,181,104,53,42,81,250,197,66,39|
84,106,149,207,209,74,73,156,81,43,107,45,99,191,145,67,84,70,101,163,84,31,121,74,75,151,220,49,197,136,45,88,72,252,144,73,140|
98,44,100,27,214,35,226,35,40,56,161,62,94,194,95,74,88,68,69,79,87,46,136,80,106,125,163,105,167,176,120,62,49,267,271,119,16|
102,124,96,148,71,74,163,231,47,31,191,78,91,171,114,124,91,84,36,100,84,49,16,79,76,111,171,80,101,220,165,136,38,185,157,215,160|
92,42,30,154,162,63,185,188,54,61,71,38,97,87,132,98,89,68,31,159,88,24,54,71,97,155,257,54,149,73,88,47,61,219,117,253,12|
89,134,161,93,101,62,250,43,36,51,184,66,99,166,111,114,85,95,41,111,91,37,93,70,110,121,173,74,152,137,170,58,42,280,88,253,55|
82,126,46,127,124,81,256,74,50,34,159,64,93,201,107,130,88,52,100,135,85,53,133,79,121,127,148,38,163,87,107,57,48,188,155,213,116|
106,104,93,98,116,41,111,59,29,29,198,63,88,79,125,68,85,87,88,69,84,34,112,73,73,152,181,68,168,140,137,77,52,79,251,209,88|
85,129,139,106,163,83,105,45,41,38,115,78,93,177,73,112,91,83,89,156,93,32,98,77,100,106,240,86,100,63,39,105,63,237,189,253,67|
101,90,116,73,90,90,216,123,25,40,93,71,92,291,84,108,82,76,95,93,93,45,52,80,117,114,150,46,177,57,155,134,54,223,186,55,29|
89,114,143,27,91,63,160,226,45,44,40,63,91,195,117,92,84,68,35,97,91,31,65,71,71,110,147,47,93,123,54,90,90,198,219,162,96|
90,58,66,198,122,77,77,109,47,62,149,57,96,223,126,77,89,69,28,158,86,20,58,71,117,153,216,91,107,88,157,110,83,87,96,253,22|
77,60,33,93,202,86,176,76,75,37,73,35,95,99,130,89,88,70,98,186,86,58,135,78,108,121,98,45,111,128,109,110,33,145,102,219,162|
85,138,114,208,105,33,251,244,35,41,50,31,88,173,139,105,90,52,79,125,89,38,14,79,111,120,121,42,92,18,79,120,59,208,86,252,63|
79,147,42,129,135,68,226,95,47,26,149,60,98,281,81,117,85,94,69,85,83,44,61,75,71,103,161,80,188,17,121,60,47,232,162,229,122|
106,21,136,111,200,50,245,136,69,47,180,69,93,252,78,94,88,62,94,194,92,24,87,71,88,102,153,66,117,123,171,107,57,171,110,162,46|
97,112,37,196,170,72,262,111,68,31,160,64,99,237,71,133,88,78,30,112,88,48,71,72,81,129,164,46,180,112,39,87,63,206,285,159,79|
105,49,95,205,126,94,135,157,70,27,78,32,98,110,106,131,90,120,57,168,87,30,145,76,102,104,137,103,140,209,125,124,97,72,234,192,17|
94,41,40,18,76,82,220,180,68,49,68,65,95,160,98,82,88,47,51,86,87,26,116,74,114,157,106,71,119,117,150,124,95,117,85,58,10|
86,39,107,61,169,42,186,51,36,28,89,72,93,248,91,129,83,101,89,173,90,54,136,80,82,93,213,67,100,153,123,102,44,182,248,104,99|
83,134,30,36,139,70,234,223,43,63,192,57,97,214,73,119,82,90,45,65,91,37,62,75,73,144,248,80,162,110,40,47,97,206,175,198,44|
81,79,35,118,165,84,105,184,77,38,61,70,99,191,106,93,83,96,95,77,92,50,65,71,87,114,199,58,89,61,70,125,46,175,268,53,70|
79,51,64,92,165,74,228,155,22,50,193,54,90,92,140,77,81,115,47,177,83,57,82,74,94,160,110,104,97,50,164,55,43,202,159,194,157|
102,148,126,185,162,59,248,148,55,56,128,61,99,82,132,129,80,97,55,126,85,26,131,76,99,104,98,69,163,45,98,78,54,95,87,80,117|
97,25,46,153,205,37,109,61,20,54,104,43,98,211,111,89,92,84,36,175,85,34,72,71,91,144,148,108,178,197,174,83,97,99,119,63,168|
90,38,144,34,182,53,248,134,46,32,201,81,97,247,124,135,90,116,63,177,90,42,126,71,123,90,100,105,87,66,149,130,79,102,80,151,90|
92,139,147,71,191,93,241,93,29,31,71,79,88,254,138,98,92,54,55,160,87,30,76,71,114,132,154,56,160,113,172,124,46,115,169,225,86|
84,155,32,145,69,50,116,149,79,63,97,94,90,276,132,117,82,88,80,191,92,24,132,78,97,96,245,68,150,62,86,143,98,242,131,194,46|
95,45,52,123,194,92,147,142,37,41,145,78,97,273,79,80,81,63,99,90,83,45,126,78,90,93,117,76,116,94,61,47,43,132,244,74,45|
91,27,136,197,170,32,93,86,65,32,90,80,98,189,79,112,82,106,74,82,84,39,58,80,82,156,88,90,116,195,110,99,82,86,151,132,109|
99,90,66,123,116,85,131,250,66,35,186,57,88,141,121,72,84,87,85,77,91,35,80,72,87,112,191,46,140,150,172,44,68,134,171,154,120|
85,135,101,24,114,86,214,192,66,52,147,92,98,237,104,94,91,49,82,178,90,25,17,70,96,97,221,39,113,132,139,89,82,106,247,148,13|
93,149,57,81,184,37,220,56,44,53,124,35,91,234,128,130,90,59,40,143,85,47,55,76,82,130,106,73,149,199,174,48,53,183,133,174,41|
95,33,73,175,131,74,205,141,46,48,80,44,93,205,117,68,81,115,82,130,89,28,97,77,84,134,125,106,152,26,70,81,90,252,263,213,83|
87,82,113,123,209,44,149,72,83,48,43,64,97,287,107,99,84,111,26,136,92,43,93,77,100,112,182,68,89,120,74,115,38,151,149,147,92|
96,87,33,183,171,34,101,217,52,28,75,49,91,178,126,131,80,72,72,75,92,27,66,73,115,103,97,37,162,29,63,81,54,282,236,153,102|
102,63,128,130,183,55,217,181,39,42,151,33,90,229,117,70,89,117,100,75,85,28,102,74,122,134,262,51,134,84,81,84,89,171,154,48,58|
83,20,51,89,182,52,87,149,21,29,200,32,88,228,122,73,85,108,59,198,89,43,84,76,79,124,231,107,96,79,73,100,96,285,279,67,24|
97,128,25,165,141,93,75,162,58,29,63,23,92,80,143,93,81,75,41,123,94,44,67,73,118,113,106,39,197,41,74,73,74,69,99,121,126|
85,56,38,173,148,48,136,167,53,42,187,41,93,123,69,118,92,118,83,160,92,51,49,70,95,150,165,106,133,213,83,130,68,258,160,227,118|
106,72,70,200,190,82,218,219,59,48,160,52,93,166,84,84,92,122,52,142,87,55,134,75,75,160,257,58,132,89,76,46,80,193,95,71,57|
89,57,80,51,123,75,67,117,40,36,76,67,91,189,145,114,92,99,98,151,87,50,135,76,88,151,137,102,134,187,44,48,40,244,203,258,16|
107,118,126,25,165,42,232,25,46,39,130,29,90,158,137,134,89,108,70,152,83,43,131,72,94,140,245,77,120,219,97,117,60,127,277,68,158|
97,81,25,169,185,35,149,187,59,64,146,88,99,249,132,133,88,103,32,183,93,31,99,77,126,159,263,77,100,62,173,79,61,161,194,247,87|
107,58,42,110,158,66,257,79,71,28,45,37,91,133,97,66,92,96,99,74,88,33,135,76,114,122,241,67,100,96,108,116,52,218,176,234,26|]; 
array [int,int] of SmartTableOp: signs0=[|Int,All,LtInt,All,Int,Int,GtInt,Int,Int,Int,GtInt,Int,Int,LtInt,Int,LtInt,GtInt,GtInt,LtInt,GtInt,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt,GtInt,LtInt,Int,Int,Int,Int|Int,LtInt,Int,Int,Int,GtInt,Int,LtInt,LtInt,Int,Int,Int,LtInt,Int,LtInt,GtInt,Int,Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,GtInt,Int,LtInt,LtInt,GtInt,GtInt,Int,Int,Int,LtInt,Int,Int|Int,GtInt,GtInt,Int,Int,Int,Int,GtInt,Int,Int,All,Int,LtInt,Int,Int,Int,Int,Int,GtInt,LtInt,GtInt,Int,All,Int,Int,Int,Int,Int,LtInt,LtInt,LtInt,LtInt,LtInt,Int,All,Int,Int|Int,All,All,Int,Int,Int,GtInt,Int,LtInt,Int,All,All,GtInt,Int,GtInt,Int,LtInt,Int,Int,Int,LtInt,Int,LtInt,Int,LtInt,LtInt,Int,Int,Int,All,Int,Int,All,Int,All,LtInt,Int|LtInt,GtInt,Int,GtInt,Int,All,Int,Int,Int,All,GtInt,GtInt,Int,All,LtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,LtInt,All,Int,Int,Int,Int,Int,Int|GtInt,Int,All,Int,GtInt,Int,Int,All,GtInt,Int,LtInt,Int,All,Int,Int,Int,GtInt,LtInt,Int,All,Int,All,Int,Int,GtInt,Int,LtInt,LtInt,Int,Int,Int,All,Int,Int,LtInt,GtInt,Int|Int,All,Int,LtInt,All,Int,GtInt,Int,Int,GtInt,All,All,Int,Int,Int,GtInt,All,Int,Int,LtInt,LtInt,LtInt,GtInt,Int,LtInt,GtInt,LtInt,LtInt,Int,Int,Int,GtInt,Int,GtInt,Int,LtInt,Int|All,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,LtInt,Int,All,Int,All,Int,GtInt,Int,Int,Int,All,LtInt,LtInt,GtInt,Int,Int,Int,LtInt,LtInt,LtInt,Int,Int,Int,GtInt,Int,All|Int,Int,All,LtInt,Int,All,All,Int,GtInt,GtInt,Int,Int,GtInt,Int,Int,Int,All,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int,All,LtInt,Int,Int,Int,GtInt,All,LtInt,Int,Int|Int,GtInt,Int,All,LtInt,GtInt,GtInt,Int,Int,Int,GtInt,Int,Int,Int,All,GtInt,Int,GtInt,LtInt,Int,All,LtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,GtInt,GtInt,Int,Int,GtInt,Int|Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int,All,GtInt,Int,LtInt,Int,Int,LtInt,GtInt,LtInt,Int,Int,GtInt,GtInt,LtInt,GtInt,Int,GtInt,Int,GtInt,Int,Int,GtInt,All,Int,LtInt,GtInt,Int,All|Int,Int,GtInt,Int,Int,Int,Int,GtInt,All,GtInt,All,Int,Int,Int,Int,GtInt,All,GtInt,LtInt,GtInt,Int,Int,All,GtInt,Int,All,Int,All,All,Int,LtInt,Int,Int,All,Int,Int,GtInt|All,LtInt,LtInt,All,LtInt,Int,Int,LtInt,Int,LtInt,GtInt,Int,Int,All,Int,Int,LtInt,LtInt,Int,All,Int,Int,LtInt,LtInt,LtInt,LtInt,All,Int,All,Int,GtInt,All,Int,Int,LtInt,Int,Int|All,Int,Int,LtInt,Int,Int,Int,All,Int,Int,Int,Int,LtInt,GtInt,Int,Int,LtInt,GtInt,GtInt,All,Int,Int,Int,GtInt,LtInt,GtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int,All,Int,Int|Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,GtInt,LtInt,All,LtInt,LtInt,Int,GtInt,Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int,Int,All,Int,Int,Int,LtInt,Int|Int,LtInt,GtInt,All,LtInt,Int,GtInt,Int,All,GtInt,GtInt,GtInt,Int,Int,Int,LtInt,LtInt,Int,Int,Int,LtInt,GtInt,Int,Int,Int,Int,LtInt,Int,LtInt,GtInt,GtInt,Int,Int,Int,Int,GtInt,All|Int,LtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,All,GtInt,All,Int,Int,Int,Int,LtInt,LtInt,Int,GtInt,Int,Int,All,GtInt,All,Int,GtInt,Int,Int,All,GtInt,Int,Int|GtInt,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,GtInt,GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,All|Int,Int,Int,Int,Int,GtInt,Int,GtInt,LtInt,LtInt,LtInt,LtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,LtInt,Int,Int,All,Int,Int,LtInt,All,Int,Int,All,Int,GtInt,Int,GtInt,Int,GtInt|GtInt,All,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,Int,LtInt,Int,GtInt,GtInt,All,Int,Int,Int,Int,Int,GtInt,Int,LtInt,All,Int,GtInt,Int,Int,GtInt,All|Int,GtInt,All,GtInt,Int,Int,Int,Int,GtInt,All,Int,Int,Int,GtInt,Int,LtInt,All,LtInt,LtInt,Int,GtInt,Int,GtInt,All,Int,LtInt,All,All,GtInt,GtInt,Int,LtInt,Int,LtInt,Int,All,Int|Int,All,Int,Int,Int,Int,GtInt,Int,Int,GtInt,Int,GtInt,LtInt,LtInt,Int,Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,All,Int,Int,Int,Int,Int,Int,Int,LtInt|Int,LtInt,GtInt,Int,LtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,Int,LtInt|Int,LtInt,Int,LtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,GtInt,Int,GtInt,Int,LtInt,GtInt,All,GtInt,Int,GtInt,GtInt,Int,LtInt,Int,LtInt,Int,Int,Int,GtInt,LtInt,Int,LtInt,Int,LtInt,Int|LtInt,Int,Int,All,Int,GtInt,GtInt,Int,Int,LtInt,Int,LtInt,GtInt,Int,GtInt,All,Int,Int,Int,All,Int,LtInt,Int,Int,Int,Int,Int,Int,LtInt,Int,LtInt,All,GtInt,Int,GtInt,Int,Int|GtInt,Int,Int,Int,Int,Int,GtInt,Int,All,Int,Int,Int,Int,All,Int,Int,All,GtInt,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,All,Int,Int,Int,Int,Int,All,GtInt|Int,Int,Int,GtInt,LtInt,Int,GtInt,Int,LtInt,LtInt,Int,Int,LtInt,Int,All,Int,LtInt,All,Int,GtInt,Int,Int,Int,All,Int,GtInt,LtInt,Int,Int,LtInt,GtInt,Int,Int,Int,Int,Int,Int|Int,Int,Int,Int,Int,GtInt,LtInt,GtInt,GtInt,Int,Int,Int,Int,All,Int,All,Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,All,All,Int,GtInt,Int,Int|Int,GtInt,GtInt,All,All,LtInt,GtInt,LtInt,Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,LtInt,LtInt,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,LtInt,LtInt,All,All,Int,Int,Int|Int,Int,Int,Int,Int,LtInt,GtInt,All,Int,GtInt,Int,Int,Int,LtInt,Int,All,Int,Int,Int,GtInt,Int,LtInt,Int,Int,GtInt,Int,GtInt,All,LtInt,LtInt,Int,Int,All,Int,All,GtInt,Int|Int,Int,LtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,Int,Int,LtInt,Int,Int,LtInt,All,Int,Int,GtInt,Int,Int,Int,Int,Int,All,GtInt,GtInt,Int,LtInt,Int,Int,GtInt|Int,LtInt,Int,All,LtInt,LtInt,Int,Int,LtInt,All,All,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int,Int,Int,LtInt,LtInt,Int,Int,All,Int,LtInt,LtInt,Int,LtInt,Int,Int,Int,GtInt,Int|All,Int,Int,Int,Int,LtInt,LtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,All,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,All,All,Int,Int,Int,Int,LtInt,LtInt,Int,Int,Int,Int|LtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,LtInt,LtInt,Int,Int,All,Int,GtInt,Int,Int,Int,Int,LtInt,LtInt,Int,Int,All,GtInt,LtInt,Int,Int,Int,GtInt|GtInt,All,Int,Int,LtInt,Int,LtInt,All,Int,GtInt,Int,GtInt,LtInt,GtInt,GtInt,Int,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,All,Int,Int,All,Int,Int,Int,GtInt,LtInt,Int,GtInt,LtInt|GtInt,All,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,LtInt,Int,Int,Int,LtInt,LtInt,Int,Int,All,Int,LtInt,Int,Int,Int|Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,LtInt,All,LtInt,GtInt,Int,LtInt,GtInt,Int,LtInt,Int,GtInt,Int,Int,Int,Int,GtInt,All,GtInt,GtInt,Int,LtInt,GtInt,Int,Int|Int,All,Int,Int,GtInt,Int,GtInt,Int,LtInt,GtInt,All,Int,GtInt,Int,Int,Int,All,Int,Int,GtInt,Int,GtInt,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int,Int,Int,GtInt,Int,LtInt,Int|Int,Int,Int,Int,Int,Int,Int,LtInt,All,Int,Int,All,LtInt,All,LtInt,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,GtInt,Int,Int,All,Int,Int,All,Int,GtInt,GtInt,LtInt,Int|GtInt,Int,All,GtInt,Int,Int,Int,All,GtInt,Int,GtInt,LtInt,GtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,Int,All,GtInt,Int,Int,LtInt,Int,Int,Int,GtInt,GtInt,Int,All|Int,All,Int,GtInt,Int,Int,GtInt,LtInt,Int,Int,Int,GtInt,Int,Int,GtInt,All,LtInt,LtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,GtInt,Int,All,GtInt,Int,GtInt,Int|GtInt,Int,GtInt,LtInt,LtInt,Int,All,Int,LtInt,Int,Int,Int,LtInt,Int,Int,Int,All,Int,GtInt,Int,GtInt,Int,Int,All,LtInt,GtInt,Int,All,Int,LtInt,LtInt,Int,Int,Int,Int,Int,Int|LtInt,LtInt,All,Int,Int,Int,Int,All,GtInt,GtInt,Int,Int,LtInt,Int,Int,All,GtInt,GtInt,All,Int,Int,LtInt,GtInt,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,Int,All,Int,LtInt,Int|LtInt,GtInt,GtInt,GtInt,All,Int,All,LtInt,Int,Int,Int,Int,LtInt,Int,Int,All,Int,LtInt,All,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,Int,Int|LtInt,All,GtInt,All,Int,GtInt,Int,Int,GtInt,LtInt,LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int,GtInt,LtInt,LtInt,LtInt,Int,Int,Int,Int,LtInt,Int,Int,LtInt,Int,Int,All,LtInt|Int,Int,Int,GtInt,LtInt,GtInt,Int,Int,GtInt,Int,Int,All,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,LtInt,Int,Int,All,Int,Int,Int,GtInt,GtInt,Int,Int,Int|Int,Int,All,Int,LtInt,Int,All,LtInt,Int,GtInt,Int,All,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int,Int,LtInt,All,Int,All,Int,LtInt|Int,LtInt,Int,All,Int,Int,Int,Int,GtInt,All,Int,Int,Int,Int,GtInt,All,GtInt,Int,Int,All,Int,Int,Int,GtInt,Int,GtInt,LtInt,GtInt,Int,LtInt,Int,Int,Int,Int,LtInt,Int,Int|Int,Int,GtInt,Int,Int,All,LtInt,Int,Int,Int,All,GtInt,Int,GtInt,LtInt,LtInt,Int,All,LtInt,GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int|Int,LtInt,Int,All,LtInt,LtInt,GtInt,Int,LtInt,GtInt,GtInt,Int,LtInt,LtInt,Int,LtInt,LtInt,All,Int,Int,GtInt,Int,Int,Int,Int,GtInt,LtInt,Int,Int,GtInt,GtInt,Int,GtInt,Int,Int,Int,Int|All,Int,Int,GtInt,Int,Int,Int,LtInt,All,Int,Int,GtInt,GtInt,Int,Int,Int,LtInt,Int,All,Int,Int,LtInt,GtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,All|All,Int,All,Int,Int,GtInt,Int,Int,GtInt,Int,GtInt,Int,Int,Int,Int,LtInt,Int,All,Int,Int,Int,GtInt,Int,Int,GtInt,Int,All,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,GtInt|All,GtInt,All,GtInt,GtInt,Int,GtInt,LtInt,Int,Int,Int,GtInt,LtInt,Int,Int,LtInt,GtInt,GtInt,GtInt,Int,Int,LtInt,LtInt,Int,Int,Int,Int,GtInt,LtInt,Int,GtInt,Int,Int,All,Int,Int,Int|GtInt,Int,Int,Int,All,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,LtInt,GtInt,GtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,GtInt,Int,Int,LtInt,LtInt,All,All,All,Int,All,All|GtInt,Int,Int,LtInt,LtInt,Int,LtInt,Int,LtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,All,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,GtInt,Int,LtInt,GtInt,Int,LtInt,Int|Int,GtInt,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,All,GtInt,LtInt,Int,Int,Int,Int,GtInt,Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int,LtInt,GtInt,Int,All,Int,Int,Int,LtInt,Int|Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,Int,Int,Int,Int,Int,All,GtInt,GtInt,GtInt,All,Int,Int,GtInt,Int,All,Int,Int,All,GtInt,Int,GtInt,Int,GtInt,LtInt,GtInt,All|Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,All,Int,LtInt,LtInt,GtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,GtInt,Int|GtInt,Int,All,Int,Int,Int,Int,Int,GtInt,Int,Int,LtInt,Int,LtInt,LtInt,Int,Int,Int,GtInt,GtInt,All,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int|All,Int,GtInt,LtInt,Int,LtInt,GtInt,Int,Int,LtInt,All,Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int,GtInt,LtInt,LtInt,LtInt,Int,Int,All,Int,All,LtInt,Int,Int,Int,Int|Int,Int,Int,LtInt,LtInt,All,Int,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,LtInt,LtInt,All,GtInt,Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int,Int|]; 
constraint smart_table([x49,x31,x28,x29,x34,x32,x50,x22,x48,x12,x38,x1,x40,x4,x37,x21,x43,x8,x17,x15,x18,x45,x3,x36,x41,x52,x7,x39,x6,x27,x30,x0,x9,x11,x42,x46,x54],t0,signs0)::uniud;
constraint not(x49=83);
constraint x31<159;
constraint not(x28=132);
constraint x29<106;
constraint not(x34=107);
constraint x32=85;
constraint not(x50=261);
constraint x22<154;
constraint x48<59;
constraint x12<33;
constraint not(x38=151);
constraint not(x1=40);
constraint not(x40=88);
constraint not(x4=227);
constraint not(x37=109);
constraint not(x21=115);
constraint not(x43=92);
constraint x8=89;
constraint x17>84;
constraint not(x15=87);
constraint x18=93;
constraint x45=34;
solve satisfy;