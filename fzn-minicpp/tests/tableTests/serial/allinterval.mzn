include "alldifferent.mzn";
include "table.mzn";
include "minicpp.mzn";

%use n=7 for sat, n=8 for unsat

int: n;

array[1..n] of var 1..n: x;      % sequence of numbers
array[1..n-1] of var 1..n-1: u;  % sequence of differences

constraint alldifferent(x);
constraint alldifferent(u);
constraint forall(i in 1..n-1)(u[i] = abs(x[i+1] - x[i]));


array [int,int] of 1..n: t = [|1,7,2,6|];
constraint table([x[1],x[2],x[3],x[4]], t) ::uniud;

solve :: int_search(x, first_fail, indomain_min, complete)
      satisfy;
