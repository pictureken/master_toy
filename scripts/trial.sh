trial_list=($(seq 1 1 5))
hidden_size_list=(1000 5 10 15 17 20 22 35 75 100)

for hidden_size in "${hidden_size_list[@]}"
do
    for trial in "${trial_list[@]}"
    do
        info="hidden size:${hidden_size} trial:${trial}"
        echo ${info}
        python train.py \
        --hidden_size ${hidden_size} \
        --epoch 500 \
        --trial ${trial}
    done
done