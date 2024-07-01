cd bin
# mpirun -np 2 gnome-terminal -- bash -c "gdb -ex run --args xhpcg 16 16 16; exec bash"
# mpirun -np 2 xterm -hold -e gdb -ex run --args xhpcg 16 16 16
mpirun -np 2 konsole -e bash -c "gdb --args xhpcg 16 16 16; exec bash"

# not run immediately
