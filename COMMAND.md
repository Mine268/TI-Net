main_pretrain.py

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 pretrain.py \
	--batch_size 108 \
	--epochs 100 \
	--warmup_epochs 40 \
	--accum_iter 1 \
	--model ae_vit_base_patch16_dec512d8b \
	--norm_pix_loss \
	--blr 1.5e-4 --weight_decay 0.05 \
	--data_path data/Imagenet-1K \
	--output_dir logs/20240812 \
	--log_dir logs/20240812
```
