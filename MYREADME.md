``` 
python test_resfusion_restore.py \
    --dataset Raindrop \
    --data_dir ../datasets/Raindrop \
    --model_ckpt "my_resfusion_train/lightning_logs/version_0/checkpoints/last.ckpt" \
    --T 12 \
    --denoising_model RDDM_Unet \
    --set_float32_matmul_precision_high

python test_adjscc_signal_resfusion.py \
    --resfusion_ckpt my_resfusion_train/lightning_logs/version_0/checkpoints/last.ckpt
    --input_dir ../datasets/Raindrop/test_a/gt \
    --output_dir my_resfusion_eval_powernorm
```
 python -m pip uninstall -y albucore simsimd stringzilla
  python -m pip check