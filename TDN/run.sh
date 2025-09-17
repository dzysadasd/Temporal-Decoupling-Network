export CUDA_VISIBLE_DEVICES=0
model_name=TDN

root_path_name=../../../dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1


model_type='linear'
seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2025
do
      root_path_name=datasets/ETTh1
      data_path_name=ETTh1.csv
      model_id_name=ETTh1
      data_name=ETTh1
      python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --period_len 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed

      root_path_name=datasets/ETTh2
      data_path_name=ETTh2.csv
      model_id_name=ETTh2
      data_name=ETTh2
      python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --period_len 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed

      root_path_name=datasets/ETTm1
      data_path_name=ETTm1.csv
      model_id_name=ETTm1
      data_name=ETTm1
      python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --period_len 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed

      root_path_name=datasets/ETTm2
      data_path_name=ETTm2.csv
      model_id_name=ETTm2
      data_name=ETTm2
      python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --period_len 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed

      root_path_name=datasets/Electricity
      data_path_name=Electricity.csv
      model_id_name=Electricity
      data_name=Electricity
      python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --period_len 168 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed $random_seed

      root_path_name=datasets/traffic
      data_path_name=traffic.csv
      model_id_name=traffic
      data_name=traffic
      python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --period_len 168 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed $random_seed

      root_path_name=../../../datasets/Weather
      data_path_name=Weather.csv
      model_id_name=Weather
      data_name=Weather
      python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --period_len 144 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed

done
done


