hidden_size_list=(1000 5 10 15 17 20 22 35 75 100)

for hidden_size in "${hidden_size_list[@]}"
do
    python tta.py \
    --hidden_size ${hidden_size} \
    --sigma 0.5
done