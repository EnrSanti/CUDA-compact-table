%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 85..264 : x0;
var 76..93 : x1;
var 85..249 : x2;
var 84..209 : x3;
var 11..197 : x4;
var 37..76 : x5;
var 68..163 : x6;
var 74..217 : x7;
var 39..266 : x8;
var 16..115 : x9;
var 1..86 : x10;
var 74..117 : x11;
var 60..72 : x12;
var 72..143 : x13;
var 8..253 : x14;
var 74..261 : x15;
var 2..208 : x16;
var 69..232 : x17;
var 60..102 : x18;
var 20..86 : x19;
var 8..166 : x20;
var 44..205 : x21;
var 8..222 : x22;
var 74..256 : x23;
var 70..133 : x24;
var 86..318 : x25;
var 19..218 : x26;
var 59..178 : x27;
var 69..233 : x28;
var 33..204 : x29;
array [int,int] of 1..318 : t0=[|192,129,154,221,112,113,130,114,84|
92,130,93,198,90,110,108,90,111|
188,69,109,214,97,111,84,152,105|
197,82,182,153,123,17,89,131,113|
90,129,68,137,125,50,174,111,98|
137,79,73,141,107,81,200,81,81|
145,135,150,116,80,44,170,160,90|
191,162,28,201,127,84,125,147,75|
187,69,159,71,82,17,215,55,75|
172,82,124,219,79,53,99,131,93|
88,140,132,208,103,52,222,136,79|
178,109,112,128,100,96,212,59,111|
163,163,158,160,116,62,122,149,75|
94,121,82,230,129,86,191,192,81|
130,68,206,177,124,37,196,86,105|
206,87,107,89,127,102,112,132,94|
135,81,140,171,82,92,168,56,102|
178,70,216,100,96,66,176,86,82|
115,110,29,201,128,56,232,120,109|
159,104,205,231,109,20,128,135,117|
101,121,101,82,123,76,159,45,106|
131,131,93,182,138,50,202,99,110|
196,99,184,178,84,92,113,104,82|
206,127,84,147,116,65,179,82,105|
153,68,34,208,126,81,86,87,99|
131,135,66,90,112,25,186,60,99|
194,150,169,113,123,92,152,86,103|
179,82,143,209,85,97,165,141,115|
107,84,35,232,137,49,100,137,92|
128,105,128,199,80,41,104,81,98|
118,121,105,143,112,43,176,153,94|
149,154,69,226,100,37,215,87,104|
135,139,199,198,75,46,188,87,74|
187,131,47,143,99,18,195,127,100|
148,105,105,213,78,59,79,57,86|
183,113,178,212,119,71,140,167,96|
189,71,29,153,91,45,89,125,83|
145,139,101,121,143,64,186,70,105|
127,163,123,203,114,32,97,83,99|
143,150,76,156,81,22,180,177,84|
85,74,156,168,126,96,119,97,78|
155,131,114,83,106,99,135,199,102|
92,108,99,179,89,105,152,139,91|
184,92,194,85,101,18,77,54,74|
193,80,160,168,136,42,145,151,110|
163,146,116,169,130,16,75,120,104|
154,98,172,73,79,31,147,140,97|
150,97,184,145,123,22,161,57,91|
148,88,126,219,89,95,228,195,97|
138,132,189,114,93,16,172,109,78|
198,157,161,101,85,50,121,124,83|
190,141,150,138,127,103,117,192,100|
178,116,36,178,98,17,96,111,102|
113,94,167,219,114,94,87,157,98|
192,101,105,109,109,27,209,87,112|
112,113,183,107,134,101,135,107,96|
145,107,75,149,100,82,189,80,98|
162,71,60,172,74,27,208,94,75|
199,100,192,212,88,22,224,81,75|
132,153,23,151,85,47,150,46,90|
156,80,26,199,95,54,218,205,113|
152,155,167,87,136,31,109,74,94|
90,137,190,119,83,39,221,67,106|
124,92,165,170,101,72,199,128,80|
142,125,78,205,135,68,202,131,79|
209,99,108,88,110,86,210,144,88|
142,104,127,216,120,93,128,156,93|
122,118,102,120,109,45,233,114,93|
172,79,200,82,134,35,75,154,99|
167,95,192,219,99,87,111,62,75|
92,119,120,148,91,83,149,197,84|
105,146,135,151,84,23,171,57,84|
170,110,174,162,134,41,188,135,113|
151,157,204,182,104,17,154,177,91|
99,76,123,76,132,35,164,93,99|
103,76,148,150,132,78,177,89,79|
149,162,54,227,102,52,220,119,107|
116,160,118,231,97,19,170,59,114|
125,115,61,169,88,63,124,155,113|
189,155,127,102,134,105,72,196,82|
121,109,198,77,141,39,141,152,80|
196,125,20,131,99,90,123,62,104|
143,70,41,116,111,71,202,94,116|
209,81,169,116,126,89,111,134,105|
167,79,143,148,110,44,103,93,102|
164,155,86,103,117,58,121,181,91|
116,79,79,191,126,73,198,94,111|
90,126,29,139,140,113,150,179,74|
88,103,116,188,99,49,189,139,109|
152,143,90,153,93,16,216,189,81|
121,151,150,92,78,92,176,170,104|
176,91,40,93,110,93,74,183,117|
124,163,75,100,119,56,159,168,100|
115,73,56,149,107,30,144,50,108|
153,70,190,212,93,93,172,132,74|
191,110,42,175,141,25,75,149,84|
119,145,137,215,103,72,208,77,112|
122,98,51,112,84,59,70,147,90|
202,84,24,211,103,59,194,148,95|
130,78,156,131,74,93,213,118,88|
103,70,91,113,131,96,223,92,95|
154,101,111,173,104,63,140,179,115|
95,120,115,106,93,63,106,167,85|
96,72,156,84,125,98,171,173,109|]; 
array [int,int] of SmartTableOp: signs0=[|GtInt,Int,GtInt,Int,Int,Int,Int,Int,Int|LtInt,Int,Int,GtInt,GtInt,Int,Int,GtInt,Int|GtInt,Int,Int,LtInt,LtInt,GtInt,LtInt,LtInt,LtInt|Int,Int,Int,GtInt,Int,LtInt,Int,Int,GtInt|Int,Int,Int,Int,Int,GtInt,All,GtInt,Int|Int,GtInt,Int,GtInt,All,Int,LtInt,Int,Int|Int,Int,Int,All,Int,Int,Int,All,Int|Int,GtInt,Int,LtInt,LtInt,Int,Int,LtInt,LtInt|Int,Int,Int,GtInt,Int,All,Int,All,Int|Int,Int,Int,LtInt,GtInt,Int,All,Int,All|Int,Int,LtInt,GtInt,GtInt,Int,Int,LtInt,All|Int,Int,Int,Int,Int,LtInt,GtInt,Int,All|Int,Int,Int,GtInt,Int,GtInt,GtInt,Int,LtInt|Int,All,Int,Int,All,Int,Int,Int,GtInt|Int,Int,GtInt,Int,Int,Int,LtInt,Int,Int|Int,Int,Int,GtInt,Int,Int,Int,Int,Int|Int,Int,Int,GtInt,Int,Int,Int,Int,Int|Int,Int,Int,LtInt,All,Int,Int,Int,Int|Int,Int,All,LtInt,Int,Int,Int,Int,All|LtInt,All,LtInt,Int,Int,Int,Int,Int,All|Int,GtInt,Int,Int,All,GtInt,Int,All,Int|Int,Int,Int,Int,Int,Int,Int,Int,Int|LtInt,Int,Int,Int,GtInt,Int,Int,LtInt,LtInt|GtInt,GtInt,Int,GtInt,Int,Int,GtInt,Int,GtInt|All,Int,GtInt,Int,Int,GtInt,Int,All,Int|LtInt,Int,All,GtInt,LtInt,LtInt,Int,LtInt,All|Int,Int,All,Int,LtInt,Int,Int,Int,Int|Int,LtInt,LtInt,Int,GtInt,LtInt,Int,Int,Int|Int,Int,GtInt,GtInt,GtInt,LtInt,Int,Int,Int|Int,Int,Int,Int,LtInt,Int,Int,Int,Int|Int,Int,LtInt,GtInt,Int,Int,Int,Int,Int|LtInt,LtInt,Int,Int,GtInt,GtInt,All,Int,Int|LtInt,GtInt,Int,LtInt,Int,Int,Int,LtInt,GtInt|Int,Int,Int,All,Int,Int,Int,Int,Int|Int,Int,Int,Int,All,Int,All,All,Int|LtInt,GtInt,Int,All,Int,LtInt,Int,Int,GtInt|All,Int,All,Int,Int,Int,Int,All,All|GtInt,Int,Int,All,LtInt,Int,Int,GtInt,GtInt|Int,Int,Int,Int,Int,Int,All,Int,GtInt|LtInt,Int,Int,Int,Int,Int,Int,Int,GtInt|Int,Int,Int,All,Int,Int,Int,Int,LtInt|Int,All,Int,Int,Int,Int,Int,Int,Int|Int,Int,Int,Int,Int,Int,LtInt,All,Int|Int,Int,Int,GtInt,Int,GtInt,Int,LtInt,Int|Int,Int,LtInt,Int,Int,GtInt,Int,LtInt,Int|GtInt,Int,Int,Int,Int,Int,GtInt,GtInt,Int|LtInt,Int,Int,Int,LtInt,LtInt,Int,Int,Int|LtInt,GtInt,Int,LtInt,Int,Int,LtInt,Int,Int|GtInt,Int,LtInt,Int,Int,GtInt,All,GtInt,Int|Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,Int,GtInt,Int,Int,Int,Int,Int,LtInt|Int,Int,Int,Int,GtInt,All,GtInt,LtInt,Int|Int,Int,All,All,Int,LtInt,Int,LtInt,GtInt|Int,LtInt,Int,Int,Int,All,LtInt,Int,Int|GtInt,Int,Int,GtInt,GtInt,GtInt,Int,GtInt,Int|Int,Int,Int,LtInt,All,Int,LtInt,LtInt,Int|Int,Int,Int,LtInt,Int,Int,Int,Int,Int|LtInt,Int,All,LtInt,Int,Int,Int,All,Int|LtInt,Int,Int,Int,Int,GtInt,Int,Int,Int|Int,LtInt,GtInt,LtInt,GtInt,Int,GtInt,Int,Int|All,Int,LtInt,GtInt,Int,Int,Int,Int,GtInt|All,Int,Int,GtInt,Int,Int,LtInt,Int,GtInt|Int,Int,Int,GtInt,Int,GtInt,Int,GtInt,Int|LtInt,GtInt,Int,Int,Int,Int,Int,Int,Int|Int,All,GtInt,Int,Int,All,GtInt,Int,LtInt|Int,Int,Int,Int,All,Int,Int,LtInt,GtInt|Int,Int,Int,Int,GtInt,All,GtInt,LtInt,GtInt|All,GtInt,LtInt,Int,Int,All,Int,Int,Int|Int,Int,Int,Int,GtInt,Int,Int,Int,Int|GtInt,LtInt,All,Int,LtInt,GtInt,Int,Int,GtInt|GtInt,GtInt,Int,Int,Int,GtInt,Int,LtInt,Int|Int,Int,All,GtInt,Int,Int,Int,All,Int|Int,All,Int,LtInt,Int,Int,Int,Int,LtInt|Int,Int,Int,GtInt,Int,GtInt,LtInt,Int,LtInt|Int,Int,Int,All,Int,Int,All,GtInt,Int|GtInt,LtInt,Int,LtInt,Int,Int,LtInt,GtInt,GtInt|Int,Int,GtInt,GtInt,Int,Int,Int,Int,Int|All,All,All,Int,GtInt,All,All,Int,Int|GtInt,Int,Int,Int,GtInt,Int,All,GtInt,LtInt|Int,All,GtInt,LtInt,Int,All,Int,LtInt,Int|Int,Int,LtInt,GtInt,Int,GtInt,Int,LtInt,GtInt|All,Int,GtInt,Int,Int,Int,Int,LtInt,Int|All,LtInt,LtInt,Int,GtInt,LtInt,Int,All,LtInt|Int,Int,Int,All,Int,Int,Int,Int,Int|Int,Int,Int,Int,GtInt,Int,Int,Int,LtInt|LtInt,GtInt,GtInt,LtInt,Int,Int,Int,GtInt,All|Int,LtInt,GtInt,All,GtInt,GtInt,LtInt,Int,GtInt|LtInt,Int,LtInt,Int,GtInt,Int,Int,Int,All|LtInt,LtInt,Int,Int,Int,GtInt,Int,Int,Int|GtInt,Int,Int,Int,Int,Int,Int,Int,GtInt|Int,All,GtInt,Int,LtInt,Int,Int,GtInt,Int|All,LtInt,Int,Int,Int,GtInt,All,LtInt,Int|LtInt,GtInt,Int,Int,Int,Int,GtInt,Int,Int|Int,GtInt,Int,Int,Int,Int,All,Int,Int|All,GtInt,Int,Int,Int,All,Int,Int,Int|LtInt,LtInt,Int,Int,Int,Int,Int,Int,GtInt|GtInt,Int,GtInt,Int,LtInt,Int,Int,Int,LtInt|Int,LtInt,Int,LtInt,Int,LtInt,Int,GtInt,Int|Int,Int,GtInt,Int,Int,GtInt,GtInt,All,LtInt|Int,Int,Int,Int,Int,Int,Int,All,Int|Int,Int,Int,Int,Int,LtInt,Int,GtInt,LtInt|Int,Int,LtInt,Int,All,Int,Int,Int,Int|Int,Int,Int,Int,Int,Int,Int,Int,Int|Int,All,Int,LtInt,Int,Int,LtInt,Int,Int|]; 
constraint smart_table([x3,x6,x26,x17,x13,x9,x28,x21,x11],t0,signs0)::gpu;
constraint not(x3=208);
constraint not(x6=80);
constraint x26>96;
constraint not(x17=211);
solve satisfy;