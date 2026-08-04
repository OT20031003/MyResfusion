``` 
python test_resfusion_restore.py \
    --dataset Raindrop \
    --data_dir ../datasets/Raindrop \
    --model_ckpt "resfusion_restore_train/lightning_logs/version_2/checkpoints/best-epoch=49-val_PSNR=24.826.ckpt" \
    --T 12 \
    --denoising_model RDDM_Unet \
    --set_float32_matmul_precision_high
```
