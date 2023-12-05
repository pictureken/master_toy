sigma_list=($(seq 0.1 0.1 1))
hidden_size_list=($(seq 1 1 100))

for hidden_size in "${hidden_size_list[@]}"
do
    for sigma in "${sigma_list[@]}"
    do
        info="hidden size:${hidden_size} sigma:${sigma}"
        echo ${info}
        python tta.py \
        --hidden_size ${hidden_size} \
        --sigma ${sigma}
    done
done