# tmux 托管可能导致无法开启训练，建议使用 screen

main_pretrain.py

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 pretrain.py \
	--batch_size 108 \
	--epochs 100 \
	--warmup_epochs 40 \
	--accum_iter 1 \
	--model vit/ae_vit_base_patch16_dec512d8b \
	--norm_pix_loss \
	--blr 1.5e-4 --weight_decay 0.05 \
	--data_path data/Imagenet-1K \
	--output_dir logs/20240812 \
	--log_dir logs/20240812
```

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/renkaiwen/.conda/envs/py310_torch113/bin/python -m torch.distributed.launch --nproc_per_node=4 pretrain.py \
        --batch_size 12 \
        --epochs 100 \
        --model vit/sl_vit_base_patch16_dec512d8b \
        --clip_grad 5.0 \
        --norm_pix_loss \
        --blr 1.5e-4 \
        --warmup_epochs 0 \
        --pin_mem \
        --output_dir logs/20240815-3 \
        --log_dir logs/20240815-3

CUDA_VISIBLE_DEVICES=2 /home/renkaiwen/.conda/envs/py310_torch113/bin/python -m torch.distributed.launch --nproc_per_node=1 pretrain.py \
	--batch_size 48 \
	--epochs 100 \
	--model vit/sl_vit_base_patch16_dec512d8b \
	--norm_pix_loss \
	--blr 1.5e-4 \
	--warmup_epochs 0 \
	--pin_mem \
	--output_dir logs/20240814 \
	--log_dir logs/20240814
```


resnet
```bash
CUDA_VISIBLE_DEVICES=6,7 /home/renkaiwen/.conda/envs/py310_torch113/bin/python -m torch.distributed.launch --nproc_per_node=2 pretrain.py \
	--batch_size 32 \
	--epochs 100 \
	--model resnet/sll_resnet50 \
	--blr 1.5e-4 \
	--warmup_epochs 1 \
	--pin_mem \
	--output_dir logs/20240816-1 \
	--log_dir logs/20240816-1
```


finetune interhand26M
```bash
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 finetune_InterHand26M.py \
	--batch_size 96 \
	--epochs 30 \
	--model resnet/pose_resnet50 \
	--backbone_ckpt ./logs/20240816-1/checkpoint-53.pth \
	--blr 1.5e-4 \
	--warmup_epochs 0 \
	--clip_grad 5.0 \
	--pin_mem \
	--output_dir logs/20240828-1 \
	--log_dir logs/20240828-1

CUDA_VISIBLE_DEVICES=7 python finetune_InterHand26M.py \
	--batch_size 192 \
	--epochs 30 \
	--model resnet/pose_resnet50 \
	--backbone_ckpt ./logs/20240816-1/checkpoint-53.pth \
	--blr 1.5e-4 \
	--warmup_epochs 0 \
	--clip_grad 5.0 \
	--pin_mem \
	--output_dir logs/20240828-1 \
	--log_dir logs/20240828-1 \
	--resume logs/20240828-1/checkpoint-3.pth
```

ft ih26m 2
```bash
CUDA_VISIBLE_DEVICES=7 nohup python finetune_InterHand26M.py \
	--batch_size 32 \
	--epochs 30 \
	--model resnet/pose_resnet50 \
	--backbone_ckpt ./logs/20240816-1/checkpoint-53.pth \
	--blr 1.5e-4 \
	--warmup_epochs 0 \
	--clip_grad 10.0 \
	--pin_mem \
	--output_dir logs/20240829-1 \
	--log_dir logs/20240829-1 > output.log 2>&1 &
```

bs=192, pid=18680
bs=32, pid 9742
