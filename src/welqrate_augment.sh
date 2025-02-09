# python main.py augment_data.split="random_cv1"
# Run first batch of commands in parallel
CUDA_VISIBLE_DEVICES=0 python main.py augment_data.split="random_cv3" &
CUDA_VISIBLE_DEVICES=0 python main.py augment_data.split="random_cv4" &
CUDA_VISIBLE_DEVICES=0 python main.py augment_data.split="random_cv5" &
CUDA_VISIBLE_DEVICES=0 python main.py augment_data.split="scaffold_seed1" &

# Wait for first batch to complete
wait

# Run remaining commands in parallel
CUDA_VISIBLE_DEVICES=0 python main.py augment_data.split="scaffold_seed2" &
CUDA_VISIBLE_DEVICES=0 python main.py augment_data.split="scaffold_seed3" &
CUDA_VISIBLE_DEVICES=0 python main.py augment_data.split="scaffold_seed4" &
CUDA_VISIBLE_DEVICES=0 python main.py augment_data.split="scaffold_seed5" &

wait
