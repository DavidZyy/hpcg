cd bin
# mpirun -np 1 xhpcg 16 16 16
mpirun -np 2 ./xhpcg 16 16 16
# mpirun -np 4 xhpcg 16 16 16
# mpirun -np 8 xhpcg 16 16 16
# mpirun -np 16 xhpcg 16 16 16
# mpirun -np 32 xhpcg 16 16 16

# mpirun -np 1 xhpcg 128 128 128
# mpirun -np 1 xhpcg 128 32 32

# ./xhpcg 16 16 16
