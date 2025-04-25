python train_LSR.py\
   -d  ./DIV2K_train_HR -d_test  ./DIV2K_valid_HR\
   --batch-size 10 --epochs 500 --val_freq 50  -lr 1e-6 --save --cuda --exp train/test_LSR --nafwidth 32\
   --mid 2  --enc 2 2 4   --dec 2 2 2   --klvl 3 --steps 4 --save_img \
   --cweight 1 --sweight 10 --pweight 0.1 --num-steps_f 16 --num-steps_r 24 --test-patch-size 1024 1024 \
   --checkpoint_f ./pretrained/lrh_f_checkpoint.pth.tar \
   --checkpoint_r ./pretrained/lrh_r_checkpoint.pth.tar \
   --checkpoint_cbd ./pretrained/dn_checkpoint.pth.tar \
   --checkpoint_lsr ./pretrained/lsr_checkpoint.pth.tar \
   --test_degrade_type 4 --finetune --test