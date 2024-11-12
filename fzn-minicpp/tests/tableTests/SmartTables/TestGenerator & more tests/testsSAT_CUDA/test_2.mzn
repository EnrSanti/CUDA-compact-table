%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 49..90 : x0;
var 38..198 : x1;
var 53..140 : x2;
var 1..103 : x3;
var 79..263 : x4;
var 53..101 : x5;
var 69..84 : x6;
var 72..207 : x7;
var 87..173 : x8;
var 52..191 : x9;
var 32..179 : x10;
var 17..141 : x11;
var 73..176 : x12;
var 18..101 : x13;
var 37..107 : x14;
var 7..242 : x15;
var 75..290 : x16;
var 82..220 : x17;
var 86..134 : x18;
var 46..226 : x19;
var 26..243 : x20;
var 81..245 : x21;
var 11..214 : x22;
var 34..128 : x23;
var 85..273 : x24;
var 51..102 : x25;
var 19..132 : x26;
var 29..277 : x27;
var 73..132 : x28;
var 64..87 : x29;
var 74..116 : x30;
var 73..256 : x31;
var 65..289 : x32;
var 19..222 : x33;
var 62..286 : x34;
var 46..110 : x35;
var 22..167 : x36;
var 44..207 : x37;
var 23..171 : x38;
var 66..153 : x39;
var 4..148 : x40;
var 81..174 : x41;
var 62..244 : x42;
var 82..208 : x43;
var 24..253 : x44;
var 90..324 : x45;
var 11..22 : x46;
var 51..84 : x47;
var 61..178 : x48;
var 29..224 : x49;
var 9..173 : x50;
var 52..62 : x51;
var 6..123 : x52;
var 38..283 : x53;
var 20..157 : x54;
var 33..186 : x55;
var 90..321 : x56;
var 36..260 : x57;
var 35..86 : x58;
var 10..129 : x59;
var 70..256 : x60;
var 75..200 : x61;
var 52..293 : x62;
var 69..228 : x63;
var 75..193 : x64;
var 81..157 : x65;
var 22..173 : x66;
var 78..226 : x67;
var 49..245 : x68;
var 79..143 : x69;
array [int,int] of 1..324 : t0=[|199,106,115,109,147,104,67,155,58,143,67,127,121,156,87,170,67,152,141,59,145|
83,62,146,51,211,136,194,95,37,99,233,132,110,36,40,250,56,19,51,81,143|
144,101,157,48,232,107,79,83,57,103,192,148,136,168,54,84,102,133,211,81,200|
214,108,57,52,133,122,290,104,206,39,232,117,144,142,77,109,140,25,109,62,140|
222,260,80,56,135,242,227,201,173,46,152,95,107,68,100,171,149,48,263,49,205|
121,221,152,109,24,228,224,131,123,277,208,169,115,124,50,135,86,181,212,69,147|
140,127,97,64,178,176,238,197,116,222,234,110,129,71,117,121,56,146,100,87,180|
113,281,151,46,102,116,274,181,149,108,179,98,142,153,58,196,171,160,172,78,116|
213,238,74,52,84,235,190,188,211,239,106,136,88,30,62,148,106,213,283,85,144|
223,169,140,104,212,234,127,213,70,35,225,199,99,26,95,152,69,166,125,67,164|
132,199,150,91,124,150,205,213,201,136,169,133,99,115,49,162,40,163,186,86,184|
96,131,41,75,177,240,236,190,13,268,225,108,109,135,107,245,85,187,226,69,198|
177,269,59,63,43,188,80,158,92,76,188,180,140,141,47,130,74,94,231,77,117|
142,89,67,72,202,79,163,216,166,49,135,168,115,91,76,180,113,137,128,51,142|
216,120,51,102,217,148,98,162,47,54,243,99,140,33,40,71,153,212,280,74,191|
79,173,107,110,164,121,182,187,120,248,228,79,88,27,75,239,157,140,233,72,123|
219,189,90,73,245,169,216,202,63,199,236,130,88,39,13,139,50,21,266,52,200|
136,137,137,105,137,165,222,211,21,35,131,180,154,26,66,113,139,89,280,50,199|
143,269,109,50,64,222,267,191,129,254,131,96,83,22,117,131,127,75,235,90,85|
193,91,139,101,66,240,86,122,168,225,100,183,127,147,93,193,63,208,229,70,200|
121,176,154,81,53,161,214,178,97,85,189,136,118,40,82,162,72,143,189,89,204|
226,281,58,47,98,162,273,98,159,229,257,109,86,135,69,215,136,194,277,71,145|
164,62,112,66,102,131,70,147,11,35,118,115,155,152,55,148,156,87,220,56,92|
79,110,36,92,142,126,246,158,130,112,240,96,112,79,17,106,94,189,237,90,199|
166,243,20,62,242,80,57,122,188,184,214,113,107,151,53,86,70,81,70,61,137|
116,91,40,78,76,153,159,155,126,208,278,152,137,91,87,82,48,159,274,61,203|
169,245,25,72,45,155,107,133,88,99,242,131,136,118,71,104,66,113,235,72,184|
196,251,140,109,111,245,138,220,85,99,92,166,142,105,48,245,152,55,66,70,185|
209,212,91,90,211,190,182,119,158,141,139,149,96,159,48,114,134,74,90,61,133|
142,102,146,76,188,150,234,215,89,91,284,94,83,36,37,191,58,221,48,84,136|
172,148,48,52,34,135,174,102,64,62,202,147,110,128,97,90,39,204,50,71,160|
115,241,124,101,120,140,66,134,68,254,132,87,154,27,38,151,161,213,87,50,121|
92,140,38,50,150,178,286,137,175,183,249,190,126,50,128,140,51,154,165,67,92|
176,239,108,90,76,230,262,178,125,270,96,147,121,82,28,237,91,133,121,49,103|
186,151,146,58,34,151,135,110,142,56,170,187,102,97,109,156,96,75,269,54,181|
143,180,41,105,159,245,286,120,67,115,103,190,139,150,117,150,42,32,59,78,104|
224,266,80,65,40,238,147,112,170,228,277,158,104,132,96,112,134,108,222,54,178|
191,156,80,98,170,262,74,118,155,117,231,175,107,127,51,139,79,116,222,58,208|
155,209,52,52,99,203,286,217,188,270,212,150,97,52,13,158,116,189,267,77,187|
121,286,25,69,71,170,229,139,70,166,166,177,120,160,106,81,84,204,201,64,85|
105,225,120,58,70,248,206,162,197,135,262,97,114,91,47,158,130,161,279,68,155|
127,271,78,92,149,127,57,102,128,82,236,191,102,131,95,223,88,129,205,51,115|
214,195,130,72,154,229,77,123,181,199,222,103,108,74,126,190,172,115,226,56,150|
189,175,122,93,34,202,233,219,70,249,111,182,119,161,92,105,132,79,47,50,191|
133,269,38,66,84,83,152,113,175,207,156,181,97,28,25,145,160,30,171,49,129|
161,116,37,71,187,151,216,210,131,224,270,160,83,99,37,240,117,31,250,56,179|
163,171,36,82,81,251,59,170,12,142,99,145,132,68,38,235,82,58,207,82,122|
224,189,136,81,193,197,250,164,171,173,147,166,129,47,86,170,114,172,187,67,94|
224,62,123,66,31,178,187,155,164,80,190,198,141,44,71,233,126,125,58,77,93|
162,83,145,70,211,233,174,93,67,94,97,124,133,143,25,157,115,186,274,87,200|
181,245,43,101,103,222,132,103,107,254,267,129,136,136,43,94,133,48,197,80,128|
143,114,114,106,164,241,265,163,27,270,258,90,125,97,56,76,99,163,88,84,158|
174,217,27,50,216,103,182,201,194,55,185,84,124,104,117,100,127,166,156,58,88|
182,260,67,64,84,97,194,149,154,168,223,150,127,173,43,253,108,111,224,84,99|
81,83,26,62,236,262,287,199,43,49,114,177,116,158,55,102,122,87,101,51,124|
151,174,90,77,188,157,102,198,164,169,281,159,81,132,38,152,106,109,272,68,91|
167,112,25,74,154,155,217,208,204,101,141,134,120,166,67,242,126,89,188,55,130|
86,249,99,90,45,101,111,105,175,95,288,102,128,65,105,218,164,141,161,60,177|
141,265,141,63,60,89,131,124,99,236,177,196,147,133,117,157,112,148,206,66,127|
117,112,66,65,97,201,117,156,157,145,169,110,102,77,53,158,167,81,71,89,83|
194,172,135,58,43,89,117,86,63,60,225,176,112,28,46,156,36,184,45,80,135|
191,208,38,58,207,114,204,218,209,169,87,90,120,92,61,92,146,131,97,71,126|
134,116,29,70,43,102,211,193,172,160,80,181,111,76,43,159,80,120,168,59,161|
173,112,89,78,174,235,94,218,120,215,229,150,100,112,63,103,83,95,140,49,186|
124,154,83,106,57,211,257,160,79,259,220,79,94,151,103,218,59,141,148,72,147|
164,276,36,53,213,184,65,92,197,146,99,143,128,112,52,206,151,213,82,89,165|
161,276,44,108,228,115,107,111,21,147,132,152,144,134,93,205,99,216,125,51,153|
99,62,155,83,47,132,75,93,35,253,113,162,104,79,52,216,70,128,108,59,207|
83,87,124,62,92,141,165,137,193,130,212,153,151,170,99,228,157,52,239,84,92|
175,69,47,52,214,246,216,134,34,176,149,112,121,107,95,225,162,125,51,57,182|
159,281,50,52,103,259,75,99,171,98,280,174,84,88,15,109,82,117,47,87,176|
161,185,32,105,55,167,292,90,197,65,193,179,143,160,22,156,163,155,74,66,173|
102,114,58,46,100,132,254,139,186,56,73,144,90,41,31,126,88,111,272,67,168|
89,78,82,55,166,174,170,145,181,69,72,110,102,24,90,201,130,43,116,69,145|
199,64,138,51,118,251,72,178,149,214,141,158,97,40,19,77,36,42,165,52,194|
214,128,155,89,111,172,293,158,200,73,234,91,96,81,83,75,48,125,209,49,159|
134,165,56,48,123,196,227,126,37,47,253,175,155,136,52,246,87,151,266,59,207|
179,159,135,79,195,210,179,99,29,102,104,184,132,41,69,201,149,159,161,58,102|
83,260,38,57,193,161,190,104,41,114,260,84,99,66,70,198,66,165,221,76,131|
173,69,59,92,85,94,145,119,23,68,258,87,112,25,128,115,137,135,43,73,199|
94,135,43,104,220,113,80,172,44,159,184,83,156,83,49,153,65,42,114,69,185|
222,86,132,57,73,177,240,206,183,143,272,80,119,95,24,219,43,98,155,86,207|
119,275,73,107,237,195,174,215,129,259,204,100,126,153,89,171,66,184,161,79,182|
195,103,46,79,102,100,174,180,47,177,112,190,95,74,116,192,147,222,123,53,150|
88,261,120,93,56,92,101,157,76,42,217,100,138,104,19,114,149,191,180,49,115|
190,267,36,97,192,245,229,110,35,80,128,123,141,87,47,153,144,186,109,70,137|
89,170,91,90,134,230,67,151,41,39,65,152,110,137,107,129,109,56,241,49,105|
115,227,108,65,75,210,159,191,63,266,108,166,117,24,25,130,103,179,178,85,179|
82,280,110,99,198,246,254,172,189,259,113,200,82,156,96,196,33,217,213,78,111|
115,143,44,70,227,112,161,200,128,230,227,197,100,100,70,207,33,21,66,60,154|
81,161,129,83,59,257,212,120,184,92,189,80,127,149,70,221,75,19,265,74,121|
115,247,97,50,215,130,165,172,61,54,249,111,131,112,129,96,70,100,178,80,114|
121,250,126,64,162,251,163,145,63,266,183,185,153,122,39,168,49,55,277,77,87|
212,91,106,60,94,196,147,162,131,264,83,168,146,50,43,143,124,57,224,68,85|
182,79,86,60,30,82,118,149,91,169,176,179,108,162,100,130,176,129,270,76,187|
162,83,45,104,58,100,193,195,91,240,147,174,139,84,13,128,78,101,39,72,99|
131,148,125,50,87,155,130,195,111,236,285,167,83,173,127,201,90,187,258,64,162|
223,205,24,67,115,207,207,159,206,173,174,83,126,45,62,112,113,187,250,89,147|
112,136,100,86,78,214,255,126,98,41,242,128,118,170,11,173,141,161,126,49,88|
199,130,64,85,59,258,192,150,145,105,83,89,100,89,116,101,66,70,188,49,84|
133,193,53,104,70,108,139,197,50,192,78,78,98,153,94,79,104,187,205,53,124|
82,197,77,88,154,243,180,145,120,266,130,115,85,142,76,159,102,94,218,79,103|
135,180,150,87,190,263,159,128,68,221,175,200,117,51,103,126,33,200,205,64,178|
185,99,47,108,214,198,224,204,32,227,140,167,97,139,129,122,93,186,117,75,85|
126,198,92,99,130,127,183,193,140,235,287,85,138,31,10,99,137,35,86,75,94|
128,90,54,76,115,226,181,179,142,99,126,92,141,145,57,75,54,77,270,57,187|
215,188,111,55,213,228,139,175,111,43,147,126,132,42,48,90,173,153,167,71,130|
133,215,141,57,126,98,182,84,72,222,274,124,113,92,11,100,45,87,46,73,116|]; 
array [int,int] of SmartTableOp: signs0=[|GtInt,Int,LtInt,Int,Int,Int,Int,Int,LtInt,All,Int,Int,Int,All,Int,GtInt,Int,Int,Int,Int,Int|LtInt,LtInt,Int,GtInt,Int,All,LtInt,Int,Int,Int,Int,GtInt,All,Int,Int,Int,Int,GtInt,Int,LtInt,All|LtInt,Int,Int,GtInt,Int,LtInt,All,Int,GtInt,LtInt,Int,All,Int,LtInt,All,LtInt,LtInt,Int,Int,LtInt,Int|Int,Int,LtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,All,Int,GtInt,Int,LtInt,Int,Int,GtInt,GtInt,Int|Int,Int,LtInt,Int,Int,Int,Int,Int,All,Int,Int,Int,GtInt,Int,GtInt,Int,GtInt,Int,All,LtInt,Int|Int,Int,GtInt,GtInt,All,GtInt,LtInt,All,LtInt,Int,All,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int|Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,All,All,All,All,All,Int,GtInt,Int,Int,Int|LtInt,Int,Int,Int,All,Int,Int,Int,LtInt,Int,GtInt,LtInt,Int,LtInt,LtInt,Int,All,All,GtInt,Int,Int|Int,Int,Int,Int,Int,LtInt,Int,Int,Int,LtInt,Int,Int,Int,LtInt,LtInt,Int,Int,Int,Int,LtInt,Int|Int,LtInt,Int,Int,LtInt,GtInt,Int,Int,All,Int,Int,Int,Int,LtInt,GtInt,Int,GtInt,Int,Int,Int,Int|GtInt,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,GtInt,Int,Int,LtInt,Int,GtInt,Int,Int,Int,GtInt|Int,Int,GtInt,LtInt,Int,Int,LtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,LtInt,GtInt,All|GtInt,GtInt,GtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,LtInt|Int,Int,All,Int,GtInt,Int,All,All,GtInt,Int,GtInt,Int,Int,Int,GtInt,Int,GtInt,Int,Int,All,Int|GtInt,LtInt,Int,GtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,LtInt,LtInt,GtInt,LtInt,Int,Int,All,Int,GtInt|Int,All,All,Int,Int,Int,Int,GtInt,Int,LtInt,All,LtInt,Int,Int,All,Int,Int,Int,GtInt,Int,LtInt|Int,GtInt,Int,All,LtInt,Int,All,Int,GtInt,LtInt,All,LtInt,Int,GtInt,GtInt,Int,Int,Int,Int,LtInt,LtInt|Int,Int,GtInt,All,All,Int,Int,GtInt,Int,GtInt,Int,All,LtInt,Int,LtInt,LtInt,LtInt,Int,Int,Int,Int|Int,Int,All,Int,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int,Int,Int,Int,LtInt,Int,LtInt,Int,All|GtInt,Int,All,Int,Int,Int,GtInt,GtInt,Int,Int,Int,GtInt,Int,Int,All,GtInt,Int,Int,GtInt,Int,Int|LtInt,All,All,Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt,GtInt,All,All,Int,Int,GtInt,Int,Int,All|All,Int,Int,LtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,All,All,Int,GtInt,Int,LtInt,Int,LtInt|LtInt,LtInt,Int,Int,GtInt,Int,All,Int,Int,LtInt,Int,GtInt,Int,LtInt,All,Int,All,Int,Int,LtInt,GtInt|GtInt,Int,All,Int,Int,Int,Int,Int,Int,LtInt,Int,LtInt,Int,All,Int,Int,Int,All,LtInt,Int,LtInt|GtInt,Int,Int,Int,Int,Int,All,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,LtInt,LtInt,Int,All|Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int|Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,LtInt,LtInt,Int,Int,Int,All|GtInt,Int,GtInt,GtInt,Int,Int,Int,Int,Int,GtInt,LtInt,Int,All,All,Int,All,Int,Int,Int,GtInt,GtInt|Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,Int,LtInt,All,Int,Int,Int,All,LtInt,GtInt,GtInt,LtInt|Int,Int,Int,Int,Int,Int,All,Int,Int,LtInt,Int,Int,Int,Int,All,Int,All,All,Int,Int,Int|Int,GtInt,Int,LtInt,Int,Int,Int,LtInt,Int,LtInt,Int,All,Int,LtInt,LtInt,Int,Int,GtInt,GtInt,All,Int|Int,Int,All,GtInt,Int,All,LtInt,Int,GtInt,Int,Int,Int,Int,Int,All,Int,Int,GtInt,All,GtInt,Int|Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,All,LtInt|Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,Int,GtInt,Int,GtInt,LtInt,GtInt,Int,LtInt,Int|Int,Int,LtInt,Int,LtInt,LtInt,Int,Int,GtInt,All,Int,Int,Int,Int,LtInt,GtInt,Int,GtInt,LtInt,Int,Int|GtInt,Int,LtInt,Int,LtInt,GtInt,LtInt,GtInt,Int,Int,All,Int,GtInt,LtInt,All,LtInt,GtInt,Int,Int,Int,LtInt|LtInt,Int,Int,Int,Int,Int,GtInt,GtInt,All,Int,GtInt,Int,Int,Int,Int,All,Int,Int,GtInt,All,Int|Int,Int,Int,Int,Int,Int,Int,All,Int,GtInt,LtInt,Int,Int,LtInt,LtInt,LtInt,Int,GtInt,Int,Int,Int|All,Int,LtInt,Int,Int,GtInt,All,Int,LtInt,LtInt,LtInt,All,LtInt,GtInt,GtInt,Int,LtInt,Int,GtInt,GtInt,Int|All,Int,LtInt,GtInt,Int,LtInt,Int,Int,GtInt,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,All|Int,LtInt,GtInt,LtInt,Int,Int,LtInt,LtInt,LtInt,LtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,Int,All,LtInt,All,Int,LtInt,LtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,All,Int,GtInt,LtInt|Int,LtInt,Int,GtInt,Int,LtInt,Int,Int,LtInt,Int,All,Int,LtInt,GtInt,Int,GtInt,Int,LtInt,LtInt,Int,Int|LtInt,LtInt,GtInt,Int,Int,Int,Int,GtInt,All,GtInt,Int,Int,All,Int,Int,All,Int,LtInt,GtInt,Int,GtInt|GtInt,All,LtInt,Int,Int,GtInt,GtInt,Int,Int,GtInt,Int,Int,All,Int,Int,LtInt,GtInt,Int,GtInt,Int,All|GtInt,Int,Int,LtInt,GtInt,All,Int,Int,Int,All,LtInt,Int,Int,LtInt,Int,Int,Int,Int,Int,Int,Int|Int,Int,GtInt,GtInt,All,Int,GtInt,Int,Int,Int,GtInt,Int,Int,Int,Int,LtInt,LtInt,Int,Int,All,Int|GtInt,Int,Int,Int,GtInt,All,LtInt,Int,Int,LtInt,GtInt,LtInt,GtInt,Int,LtInt,LtInt,Int,Int,LtInt,Int,LtInt|LtInt,GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,All,Int,Int,Int,LtInt,All,LtInt,Int,Int|Int,Int,GtInt,Int,LtInt,Int,Int,Int,Int,LtInt,LtInt,Int,All,Int,Int,Int,Int,Int,Int,LtInt,Int|LtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,GtInt,All,LtInt,Int,GtInt,All,Int,LtInt,Int,Int,Int,LtInt|Int,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int|Int,Int,Int,LtInt,Int,Int,Int,Int,LtInt,Int,Int,GtInt,Int,GtInt,Int,GtInt,Int,All,Int,Int,All|Int,Int,All,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,All,Int,LtInt,LtInt,Int,All,All,GtInt,Int|Int,Int,Int,Int,GtInt,Int,GtInt,Int,GtInt,Int,Int,LtInt,GtInt,Int,Int,LtInt,Int,Int,Int,Int,Int|Int,Int,GtInt,Int,LtInt,Int,Int,Int,LtInt,Int,Int,GtInt,Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int|Int,Int,Int,GtInt,Int,All,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,All,LtInt,Int,Int|Int,LtInt,Int,Int,Int,Int,All,Int,Int,LtInt,Int,GtInt,Int,Int,LtInt,LtInt,All,All,Int,Int,Int|Int,GtInt,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,LtInt|LtInt,Int,LtInt,Int,All,LtInt,LtInt,Int,LtInt,GtInt,Int,Int,GtInt,LtInt,All,Int,GtInt,Int,GtInt,Int,GtInt|All,Int,Int,Int,Int,GtInt,LtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int,All,Int,Int,Int,LtInt,Int|Int,Int,LtInt,GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,All,Int,Int,LtInt,Int,Int|Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int,Int,All,Int,GtInt,GtInt,GtInt,GtInt,Int,Int,GtInt,Int,Int|GtInt,Int,Int,Int,Int,Int,All,All,LtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt|Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,Int,All,GtInt,Int,Int,LtInt,Int,LtInt,Int,Int|Int,Int,LtInt,Int,Int,GtInt,Int,Int,Int,GtInt,Int,All,Int,All,GtInt,Int,Int,All,GtInt,GtInt,LtInt|Int,Int,LtInt,LtInt,Int,LtInt,GtInt,GtInt,Int,Int,Int,Int,Int,GtInt,All,LtInt,All,GtInt,All,Int,GtInt|Int,Int,LtInt,Int,GtInt,LtInt,LtInt,Int,Int,GtInt,Int,Int,GtInt,Int,Int,GtInt,Int,Int,Int,LtInt,Int|LtInt,Int,Int,All,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int,GtInt,Int,All,LtInt,Int,Int|Int,Int,GtInt,LtInt,Int,GtInt,GtInt,All,LtInt,GtInt,LtInt,Int,Int,LtInt,LtInt,Int,Int,LtInt,All,GtInt,Int|All,Int,GtInt,Int,Int,LtInt,GtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,GtInt,Int,Int|Int,GtInt,Int,All,Int,Int,Int,Int,Int,All,Int,Int,Int,GtInt,Int,Int,LtInt,Int,Int,All,All|Int,GtInt,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,Int,LtInt,GtInt,Int,Int,LtInt|Int,Int,GtInt,Int,LtInt,Int,LtInt,Int,GtInt,Int,Int,GtInt,Int,Int,LtInt,Int,Int,Int,All,Int,All|LtInt,GtInt,LtInt,Int,Int,GtInt,LtInt,LtInt,Int,LtInt,All,Int,GtInt,Int,Int,Int,GtInt,LtInt,Int,Int,GtInt|Int,Int,Int,Int,Int,LtInt,Int,LtInt,LtInt,Int,GtInt,Int,Int,LtInt,Int,GtInt,Int,Int,All,Int,Int|GtInt,Int,Int,LtInt,Int,LtInt,Int,All,Int,Int,All,Int,Int,LtInt,Int,LtInt,Int,Int,Int,Int,GtInt|All,GtInt,GtInt,Int,Int,LtInt,Int,Int,All,GtInt,Int,Int,Int,All,LtInt,Int,GtInt,All,Int,Int,Int|All,Int,Int,Int,Int,LtInt,LtInt,GtInt,Int,LtInt,GtInt,Int,Int,Int,All,Int,Int,Int,Int,All,Int|GtInt,LtInt,All,All,Int,Int,GtInt,LtInt,Int,Int,Int,Int,LtInt,LtInt,Int,GtInt,Int,Int,GtInt,LtInt,Int|GtInt,Int,Int,Int,All,Int,All,Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int,Int,LtInt,Int,All,Int|Int,Int,Int,Int,GtInt,GtInt,GtInt,Int,LtInt,Int,Int,All,Int,All,GtInt,All,Int,Int,GtInt,Int,Int|Int,All,Int,Int,Int,Int,Int,Int,LtInt,All,Int,LtInt,Int,Int,GtInt,All,Int,Int,Int,GtInt,Int|LtInt,All,Int,Int,All,Int,Int,GtInt,Int,Int,Int,Int,Int,Int,LtInt,Int,Int,GtInt,GtInt,Int,Int|All,LtInt,LtInt,LtInt,Int,All,All,All,GtInt,Int,Int,All,LtInt,GtInt,Int,All,LtInt,LtInt,Int,Int,Int|Int,LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int,Int,GtInt,Int,GtInt,Int,GtInt,GtInt,Int,Int,Int,Int|Int,Int,All,LtInt,LtInt,GtInt,Int,Int,Int,Int,Int,All,LtInt,Int,Int,Int,Int,Int,Int,Int,Int|All,Int,GtInt,Int,GtInt,Int,Int,Int,LtInt,Int,LtInt,Int,GtInt,Int,LtInt,Int,Int,Int,Int,Int,Int|Int,LtInt,Int,Int,Int,LtInt,LtInt,Int,Int,Int,Int,GtInt,LtInt,Int,Int,All,LtInt,LtInt,Int,GtInt,Int|GtInt,Int,LtInt,LtInt,Int,Int,Int,Int,GtInt,GtInt,LtInt,Int,LtInt,All,Int,Int,Int,Int,GtInt,LtInt,All|Int,LtInt,LtInt,GtInt,All,Int,Int,LtInt,LtInt,Int,Int,Int,Int,Int,GtInt,All,GtInt,LtInt,Int,GtInt,Int|Int,GtInt,Int,GtInt,GtInt,LtInt,Int,GtInt,Int,Int,Int,Int,Int,Int,Int,All,Int,GtInt,GtInt,LtInt,Int|Int,Int,Int,Int,GtInt,LtInt,Int,Int,Int,Int,Int,Int,Int,LtInt,Int,All,GtInt,Int,Int,Int,Int|Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt,All,LtInt,Int,Int,Int,Int,GtInt,GtInt,Int,Int,GtInt,Int|Int,GtInt,Int,GtInt,GtInt,Int,Int,LtInt,Int,All,LtInt,Int,Int,LtInt,Int,Int,All,Int,Int,Int,Int|Int,Int,Int,Int,LtInt,All,LtInt,LtInt,Int,Int,Int,Int,Int,GtInt,Int,Int,GtInt,GtInt,LtInt,Int,All|GtInt,Int,GtInt,LtInt,All,All,GtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,Int,All,Int,GtInt,Int,LtInt|Int,Int,GtInt,Int,LtInt,GtInt,Int,All,Int,Int,Int,Int,Int,Int,Int,Int,Int,LtInt,LtInt,GtInt,Int|Int,Int,Int,LtInt,Int,GtInt,Int,GtInt,GtInt,Int,Int,All,GtInt,Int,Int,Int,Int,Int,Int,Int,Int|GtInt,All,All,GtInt,GtInt,Int,All,GtInt,All,LtInt,GtInt,Int,Int,Int,Int,Int,Int,Int,All,All,Int|GtInt,Int,Int,Int,Int,Int,Int,Int,Int,GtInt,GtInt,GtInt,Int,Int,GtInt,LtInt,All,GtInt,Int,GtInt,GtInt|GtInt,LtInt,Int,GtInt,Int,GtInt,Int,Int,LtInt,All,GtInt,LtInt,Int,LtInt,Int,Int,All,All,Int,LtInt,GtInt|LtInt,Int,Int,All,Int,Int,Int,All,Int,Int,All,Int,LtInt,Int,Int,GtInt,GtInt,Int,Int,LtInt,Int|Int,GtInt,LtInt,GtInt,LtInt,LtInt,Int,Int,Int,Int,Int,LtInt,Int,Int,Int,Int,GtInt,GtInt,Int,GtInt,GtInt|GtInt,All,GtInt,All,LtInt,All,Int,Int,GtInt,All,LtInt,Int,Int,Int,All,Int,Int,All,Int,All,Int|Int,All,Int,GtInt,Int,Int,Int,Int,Int,GtInt,GtInt,LtInt,Int,Int,GtInt,All,All,Int,GtInt,GtInt,Int|Int,Int,Int,Int,LtInt,Int,GtInt,Int,Int,GtInt,Int,Int,Int,Int,Int,GtInt,Int,GtInt,Int,Int,Int|Int,Int,Int,Int,Int,GtInt,Int,Int,All,LtInt,GtInt,Int,Int,Int,Int,LtInt,LtInt,Int,LtInt,Int,Int|]; 
constraint smart_table([x67,x34,x54,x35,x44,x4,x62,x17,x22,x27,x32,x61,x65,x66,x59,x60,x10,x33,x53,x0,x43],t0,signs0)::gpu;
constraint not(x67=111);
constraint x34>101;
constraint x54=40;
constraint not(x35=86);
constraint not(x44=141);
constraint not(x4=113);
constraint not(x62=131);
constraint not(x17=173);
constraint not(x22=166);
constraint x27>76;
constraint not(x32=82);
constraint x61<95;
constraint not(x65=95);
constraint not(x66=72);
constraint not(x59=119);
constraint x60<153;
solve satisfy;