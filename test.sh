#python train.py --name mpv_half --dataroot /data/bbangsik/VITON-HD/results/ALIAS_MPV_train_dataset  --gpu_ids "5" --niter 5 --niter_decay 5  --nThreads 4 --batchSize 1 --no_flip --continue_train 
python test.py --dataroot /data/bbangsik/ClothFlow_ALIAS_dataset/results/styleviton_fb_bot_agnostic  --gpu_ids "2,3" --no_flip --serial_batches --name ALIAS_demo
