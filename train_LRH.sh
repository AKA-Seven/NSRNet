
 CUDA_VISIBLE_DEVICES=0 python train_LRH.py \
 -d  ../PIRNet-local/DIV2K_train_HR \
 -d_test  ../PIRNet-local/DIV2K_valid_HR \
 --checkpoint_f ./pretrained/lrh_f_checkpoint.pth.tar \
 --checkpoint_r ./pretrained/lrh_r_checkpoint.pth.tar \
 --batch-size 10 -lr 1e-4  --save --save-images \
 --cuda --exp train/test_LRH --num-steps_f 16 --num-steps_r 24  --guide-weight 1 --rec-weight 2 \
 --test
