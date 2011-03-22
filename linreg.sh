#1/bin/bash
export PATH=$PATH:/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
bin/linreg/linreg $1
