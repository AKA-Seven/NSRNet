CUDA_VISIBLE_DEVICES=0 python finetune.py  -d  ./DIV2K_train_HR -d_test  ./DIV2K_valid_HR\
 --checkpoint_f ./pretrained/lrh_f_checkpoint.pth.tar \
 --checkpoint_r ./pretrained/lrh_r_checkpoint.pth.tar \
 --checkpoint_cbd ./pretrained/dn_checkpoint.pth.tar \
 --batch-size 10 -lr 2e-5 5e-5 1e-5 --epochs 1200 --val-freq 30 --patience 500 --num-steps_f 16 --num-steps_r 24 \
 --save --save-images --exp finetune  --attack_level 0.5 0.1 --random --nrate 0.4 --brate 0.3 --lrate 0.3 --loss_weights 1 0.25 2 2 \
 --finetune 5
# all