#!/bin/sh 

. /home/ac/jstuart/.bashrc
cd ~
cd gpmr
NODE_COUNT=`wc -l < $PBS_NODEFILE`
mvapich2-start-mpd
cat $PBS_NODEFILE
for num_nodes in `seq $((NODE_COUNT - 3)) $NODE_COUNT`; do
  timing_file=`printf "intcount.timing.%02d" $num_nodes`
  mpirun -machinefile $PBS_NODEFILE -n $num_nodes bin/intcount/intcount $elems > $timing_file
done
mpdallexit

