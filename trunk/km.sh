#!/bin/sh 

. /home/ac/jstuart/.bashrc
cd ~
cd gpmr
NODE_COUNT=`wc -l < $PBS_NODEFILE`
mvapich2-start-mpd
cat $PBS_NODEFILE
for num_nodes in `seq $((NODE_COUNT - 3)) $NODE_COUNT`; do
  timing_file=`printf "kmeans.timing.%02d" $num_nodes`
  mpirun -machinefile $PBS_NODEFILE -n $num_nodes bin/kmeans/kmeans > $timing_file
done
mpdallexit

