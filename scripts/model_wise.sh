hidden_size_list=($(seq 1 1 100))

for hidden_size in "${hidden_size_list[@]}"
do
    python train.py \
    --hidden_size ${hidden_size} \
    --epoch 1000 \
    --trial 1
done