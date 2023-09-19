deepspeed src/train.py \
    --method gc_bc \
    --steps 300000 \
    --warmup_steps 10000 \
    --save_dir gc_bc_save \
    --random_seed 42