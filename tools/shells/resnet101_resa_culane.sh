#!/bin/bash
# Trained weights: resnet101_resa_culane_20211016.pt
exp_name=resnet101_resa_culane
url=tcp://localhost:12345
# Training
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_landec.py --epochs=12 --lr=0.048 --warmup-steps=600 --batch-size=2 --workers=2 --dataset=culane --method=resa --backbone=resnet101 --world-size=8 --dist-url=${url} --exp-name=${exp_name}
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=8 --continue-from=${exp_name}.pt --dataset=culane --method=resa --backbone=resnet101 --exp-name=${exp_name}
# Testing with official scripts
./autotest_culane.sh ${exp_name} test
