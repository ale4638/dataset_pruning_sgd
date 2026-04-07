conda activate twist2
python scripts/run_random_pruning.py \
    --config configs/random_pruning.yaml \
    --datasets cifar50 --seeds 0 1 2 --device cuda
python scripts/run_random_pruning.py \
    --config configs/random_pruning.yaml \
    --datasets cifar100 \
    --seeds 0 1 2 \
    --device cuda