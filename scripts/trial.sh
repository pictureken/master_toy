trial_list=($(seq 1 1 1000))

for trial in "${trial_list[@]}"
do
    python train.py \
    --hidden_size 10000 \
    --epoch 1000 \
    --trial ${trial}
done