# CUDA-compact-table

The main branch integrating a table constaint in MiniCPP. 

┌───┬───┬───┐
│ x │ y │ z │
├───┼───┼───┤
│ 3 │ 1 │ 1 │
├───┼───┼───┤
│ 1 │ 2 │ 3 │
├───┼───┼───┤
│ 2 │ 3 │ 3 │
└───┴───┴───┘

The repository, has multiple branches, which implement the constraint either with domain filtering or table update on the device (alonside the test branch containing the results).

In CT^uf both filtering and update are kernels (recommended for high end GPUs).
In CT^f only the filtering procedure is a kernel (recommended for lower end GPUs).
In CT^u only the update procedure is a kernel.
