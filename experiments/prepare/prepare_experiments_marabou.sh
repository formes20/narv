#!/usr/bin/bash

# prepare 5_3 complete cegar 25000 50 and then generate other experiments from it
# its content is:
###############################################
##   #!/bin/bash`
##   #
##   #SBATCH --job-name=yizhak_test_marabou
##   #SBATCH --cpus-per-task=1
##   #SBATCH --output=/cs/labs/guykatz/yizhak333/ar_nn_experiments/exp_marabou_5_3_l250.output
##   #SBATCH --partition=long
##   #SBATCH --time=01:00:00
##   #SBATCH --signal=B:SIGUSR1@300
##
##   export PYTHONPATH=$PYTHONPATH:/cs/usr/yizhak333/Research/Code/Marabou
##   python3 /cs/usr/yizhak333/Research/Code/CEGAR_NN/one_experiment_marabou.py -f ACASXU_run2a_5_3_batch_2000.nnet -l 25000 -p
###############################################


PAERENT_DIR="/cs/labs/guykatz/yizhak333/ar_nn_experiments"
DIR=sbatch_experiments_files
cd $PAERENT_DIR
mkdir ${PAERENT_DIR}/${DIR}
cp ${PAERENT_DIR}/exp_marabou_5_3_l250.sbatch $DIR/
cd $DIR

for net_number in 1_1 1_2 1_3 1_4 1_5 1_6 1_7 1_8 1_9 \
2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8 2_9 \
3_1 3_2 3_3 3_4 3_5 3_6 3_7 3_8 3_9 \
4_1 4_2 4_3 4_4 4_5 4_6 4_7 4_8 4_9 \
5_1 5_2 5_3 5_3 5_4 5_5 5_6 5_7 5_8 5_9
do
    for lower_bound in 100 500
    do
        cp exp_marabou_5_3_l250.sbatch exp_marabou_${net_number}_l${lower_bound}.sbatch
        sed -i "s/5_3/${net_number}/g" exp_marabou_${net_number}_l${lower_bound}.sbatch
        # update lower_bound
        sed -i "s/l250/l${lower_bound}/g" exp_marabou_${net_number}_l${lower_bound}.sbatch
        sed -i "s/l\ 250/l\ ${lower_bound}/g" exp_marabou_${net_number}_l${lower_bound}.sbatch
    done
done
