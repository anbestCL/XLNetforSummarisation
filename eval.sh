#!/bin/sh
python3 eval.py \
--data_dir=dev_data_2 \
--do_evaluate=True \
--batch_size=8 \
--sum_len=56 \
--is_cuda=True \
--repetition_penalty=1.2 \
--max_seqlen=1024 > log_dev_data_2_rest_w_penalty.txt

python3 eval.py \
--data_dir=test_data_0 \
--do_evaluate=True \
--batch_size=8 \
--sum_len=56 \
--is_cuda=True \
--max_seqlen=1024 > log_test_data_0_wo_penalty.txt