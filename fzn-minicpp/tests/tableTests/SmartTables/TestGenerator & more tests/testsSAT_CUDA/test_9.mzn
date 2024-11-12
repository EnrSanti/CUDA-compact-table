%test automatically generated

include "table.mzn";
include "minicpp.mzn";

var 70..98 : x0;
var 81..182 : x1;
var 64..285 : x2;
var 12..67 : x3;
var 36..206 : x4;
var 1..21 : x5;
var 84..183 : x6;
var 66..196 : x7;
var 10..173 : x8;
var 56..139 : x9;
var 52..149 : x10;
var 36..283 : x11;
var 84..124 : x12;
var 9..81 : x13;
var 60..96 : x14;
array [int,int] of 1..285 : t0=[|136,72,136,147,14,49,107,84,174,97|
129,278,178,174,15,71,185,77,118,107|
71,220,184,241,35,68,180,82,168,102|
147,49,111,70,59,42,41,67,137,105|
138,103,87,68,27,16,109,70,164,111|]; 
array [int,int] of SmartTableOp: signs0=[|Int,Int,Int,Int,Int,LtInt,LtInt,LtInt,Int,Int|Int,Int,Int,LtInt,Int,Int,Int,GtInt,Int,Int|Int,LtInt,GtInt,Int,Int,Int,Int,All,LtInt,LtInt|LtInt,LtInt,GtInt,Int,Int,Int,Int,Int,Int,All|LtInt,Int,Int,Int,Int,Int,Int,Int,Int,All|]; 
constraint smart_table([x8,x11,x7,x2,x3,x13,x4,x14,x1,x12],t0,signs0)::gpu;
constraint not(x8=113);
constraint not(x11=178);
constraint not(x7=179);
constraint x2<127;
constraint not(x3=58);
constraint not(x13=42);
constraint not(x4=155);
constraint not(x14=91);
solve satisfy;