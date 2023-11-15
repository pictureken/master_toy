trial_list=($(seq 1 1 10))

for trial in "${trial_list[@]}"
do
    python test.py \
    --hidden_size 10000 \
    --epoch 1000 \
    --trial ${trial}
done