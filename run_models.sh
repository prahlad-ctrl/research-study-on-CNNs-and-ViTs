# running just every model on default data size and res

MODELS=("resnet18" "resnet50" "vit_tiny" "vit_small")

for MODEL in "${MODELS[@]}"; do
    python train.py --model $MODEL
done