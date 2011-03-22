#!/bin/sh 

. /home/ac/jstuart/.bashrc
cd ~
cd gpmr
cat $PBS_NODEFILE
mvapich2-start-mpd
mpirun -machinefile $PBS_NODEFILE -n 8 bin/linreg/linreg
mpdallexit

