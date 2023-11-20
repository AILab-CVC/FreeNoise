name="base_256_test"

ckpt='checkpoints/base_256_v1/model_256.pth'
config='configs/inference_t2v_tconv256_v1.0_freenoise.yaml'

prompt_file="prompts/single_prompts.txt"
res_dir="results_freenoise_single_256"

python3 scripts/evaluation/inference_freenoise.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 3 \
--bs 1 --height 256 --width 256 \
--unconditional_guidance_scale 15.0 \
--ddim_steps 50 \
--ddim_eta 0.0 \
--prompt_file $prompt_file \
--fps 8 \
--frames 128 \
--window_size 16 \
--window_stride 4 

