#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4gb:ngpus=1:accelerator_model=gtx1080ti
#PBS -l walltime=11:59:00
#PBS -A "xlnet"

set -e

##Scratch-Laufwerk definieren und erzeugen
SCRATCHDIR=/scratch_gs/schultsi/

## Log-File definieren
export LOGFILE=$SCRATCHDIR/$PBS_JOBNAME"."$PBS_JOBID".log"

module load Python/3.6.5

pip3 install --user numpy torch pytorch_transformers rouge

cd $SCRATCHDIR

python eval_dev.py \
--data_dir=dev_data_9 \
--do_evaluate=True \
--batch_size=8 \
--sum_len=67 \
--is_cuda=True \
--max_seqlen=1024>$LOGFILE
