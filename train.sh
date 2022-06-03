#python train.py --name mpv_half --dataroot /data/bbangsik/VITON-HD/results/ALIAS_MPV_train_dataset  --gpu_ids "5" --niter 5 --niter_decay 5  --nThreads 4 --batchSize 1 --no_flip --continue_train 
# ETRI BOTTOM CLOTHFLOW
#python train.py --dataroot /data/bbangsik/ClothFlow_ALIAS_dataset/results/styleviton_fb_bot_agnostic  --gpu_ids "2,3" --niter 15 --niter_decay 5  --nThreads 4 --batchSize 2 --no_flip --serial_batches --name ETRI-Bot_new_L1
# MPV TOP  PB-AFN
python train.py --checkpoints_dir /mnt/linux/linux_workspace/checkpoints\
 --dataroot dataset/use_warp \
 --gpu_ids "0" \
 --niter 15 --niter_decay 5 \
 --nThreads 6 --batchSize 1 \
 --no_flip --serial_batches \
 --name alias_bbangsik \
 --dataset_etri \
 --no_L1_loss \
 --use_warp \
 #--swap_warped_cloth
