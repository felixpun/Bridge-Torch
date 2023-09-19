CKPTS=(
    "gc_bc_save/190000/mp_rank_00_model_states.pt"
)

CONFIGS=(
    "gc_bc_save/config.json"
)

VIDEO_DIR=""

CMD="python src/eval.py \
    --num_timesteps 100 \
    --video_save_path videos/$VIDEO_DIR \
    $(for i in "${!CKPTS[@]}"; do echo "--checkpoint_path ${CKPTS[$i]} "; done) \
    $(for i in "${!CONFIGS[@]}"; do echo "--config_path ${CONFIGS[$i]} "; done) \
    --blocking \
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.1" --initial_eep "0.3 0.0 0.1"