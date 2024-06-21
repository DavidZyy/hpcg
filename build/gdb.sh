cd bin
mpirun -np 2 xterm -hold -e gdb -ex run --args xhpcg 32 24 16
