trial_list=($(seq 1 1 1000d))

for trial in "${trial_list[@]}"
do
    python train.py \
    --hidden_size 100 \
    --epoch 1 \
    --trial ${trial}
done