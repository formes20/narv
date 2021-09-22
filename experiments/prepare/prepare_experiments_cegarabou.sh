#!/usr/bin/bash

# prepare NET_NUMBER ABSTRACTION REFINEMENT LOWER_BOUND A_SEQUENCE R_SEQUENCE
# its content is:
###############################################
##   #!/bin/bash
##   #
##   #SBATCH --job-name=yizhak_test_QUERY_METHOD
##   #SBATCH --cpus-per-task=1
##   #SBATCH --output=/cs/labs/guykatz/yizhak333/ar_nn_experiments/QUERY_METHOD/exp_QUERY_METHOD_nnNET_NUMBER_aABSTRACTION_rREFINEMENT_l250_asA_SEQUENCE_rsR_SEQUENCE.output
##   #SBATCH --partition=long
##   #SBATCH --time=01:00:00
##   #SBATCH --signal=B:SIGUSR1@300
##
##   export PYTHONPATH=$PYTHONPATH:/cs/usr/yizhak333/Research/Code/Marabou
##   python3 /cs/usr/yizhak333/Research/Code/CEGAR_NN/one_experiment_QUERY_METHOD.py -f ACASXU_run2a_NET_NUMBER_batch_2000.nnet -a ABSTRACTION -r REFINEMENT -l LOWER_BOUND -p -as A_SEQUENCE -rs R_SEQUENCE
###############################################


PAERENT_DIR="/cs/labs/guykatz/yizhak333/ar_nn_experiments/cegarabou"
DIR=sbatch_experiments_files
cd $PAERENT_DIR
mkdir ${PAERENT_DIR}/${DIR}
cp ${PAERENT_DIR}/exp_cegarabou_5_3_complete_cegar_l250.sbatch $DIR/
cd $DIR

for net_number in 1_1 1_2 1_3 1_4 1_5 1_6 1_7 1_8 1_9 \
2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8 2_9 \
3_1 3_2 3_3 3_4 3_5 3_6 3_7 3_8 3_9 \
4_1 4_2 4_3 4_4 4_5 4_6 4_7 4_8 4_9 \
5_1 5_2 5_3 5_3 5_4 5_5 5_6 5_7 5_8 5_9
do
    for lower_bound in 100 500
    do
        for abstraction_type in heuristic complete
        do
            for refinement_type in cegar cetar
            do
                for ref_sequence_length in 250 500 1000
                do
                    cp exp_cegarabou_NET_NUMBER_complete_cegar_l250.sbatch exp_cegarabou_${net_number}_${abstraction_type}_${refinement_type}_l${lower_bound}_s${ref_sequence_length}.sbatch
                    sed -i "s/NET_NUMBER/${net_number}/g" exp_cegarabou_${net_number}_${abstraction_type}_${refinement_type}_l${lower_bound}_s${ref_sequence_length}.sbatch
                    sed -i "s/ABSTRACTION/${abstraction_type}/g" exp_cegarabou_${net_number}_${abstraction_type}_${refinement_type}_l${lower_bound}_s${ref_sequence_length}.sbatch
                    sed -i "s/REFINEMENT/${refinement_type} /g" exp_cegarabou_${net_number}_${abstraction_type}_${refinement_type}_l${lower_bound}_s${ref_sequence_length}.sbatch
                    # update ref_sequence_length
                    sed -i "s/s50/s${ref_sequence_length}/g" exp_cegarabou_${net_number}_${abstraction_type}_${refinement_type}_l${lower_bound}_s${ref_sequence_length}.sbatch
                    sed -i "s/s\ 50/s\ ${ref_sequence_length}/g" exp_cegarabou_${net_number}_${abstraction_type}_${refinement_type}_l${lower_bound}_s${ref_sequence_length}.sbatch
                    # update lower_bound
                    sed -i "s/l250/l${lower_bound}/g" exp_cegarabou_${net_number}_${abstraction_type}_${refinement_type}_l${lower_bound}_s${ref_sequence_length}.sbatch
                    sed -i "s/l\ 250/l\ ${lower_bound}/g" exp_cegarabou_${net_number}_${abstraction_type}_${refinement_type}_l${lower_bound}_s${ref_sequence_length}.sbatch
                    # sbatch exp_ar_${net_number}_${abstraction_type}_${refinement_type}_u${lower_bound}_sS{ref_sequence_length}.sbatch
                done
            done
        done
    done
done
