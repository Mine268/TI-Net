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

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/renkaiwen/.conda/envs/py310_torch113/bin/python -m torch.distributed.launch --nproc_per_node=4 pretrain.py \
	--batch_size 48 \
	--epochs 100 \
	--model sl_vit_base_patch16_dec512d8b \
	--norm_pix_loss \
	--blr 1.5e-4 \
	--warmup_epochs 0 \
	--pin_mem \
	--output_dir logs/20240815 \
	--log_dir logs/20240815

CUDA_VISIBLE_DEVICES=2 /home/renkaiwen/.conda/envs/py310_torch113/bin/python -m torch.distributed.launch --nproc_per_node=1 pretrain.py \
	--batch_size 48 \
	--epochs 100 \
	--model sl_vit_base_patch16_dec512d8b \
	--norm_pix_loss \
	--blr 1.5e-4 \
	--warmup_epochs 0 \
	--pin_mem \
	--output_dir logs/20240814 \
	--log_dir logs/20240814
```
