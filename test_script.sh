#!/bin/bash

gpu_id=${2:-0}
root_path_gloriphy="$(pwd)/GLoRiPHY_source"
root_path_baseline="$(pwd)/Baseline"
sim_data_dir="$1/simulation"
real_data_dir="$1/real_world"
sf_values=(8 9)
nodes=(D E F G H)

# Loop through SF8 and SF9 for GLoRiPHY simulation tests
for sf in 8 9; do
  num_packets=100
  if [ "$sf" -eq 9 ]; then
    num_packets=120
  fi

  python $root_path_gloriphy/main.py \
    --root_path "$root_path_gloriphy" \
    --train_denoiseGen \
    --data_dir "$sim_data_dir/test_SF_$sf" \
    --num_packets $num_packets \
    --checkpoint_dir testing/test_SF${sf}_sim \
    --load checkpoints/denoiseGen/simulation/SF${sf} \
    --load_symbol_conformer checkpoints/AWGN_filter/SF${sf}/best_conformer_jgj.pkl \
    --sf $sf \
    --sim_data \
    --test_mode \
    --transformer_encoder_dim 256 \
    --free_gpu_id "$gpu_id"

  python $root_path_gloriphy/offset_correction.py \
    --data_dir "$sim_data_dir/test_SF_$sf" \
    --sf $sf \
    --num_packets $num_packets \
    --sim_data
done

# Loop through Nodes D-H for GLoRiPHY real-world tests
for node in "${nodes[@]}"; do
  python $root_path_gloriphy/main.py \
    --root_path "$root_path_gloriphy" \
    --train_denoiseGen \
    --data_dir "$real_data_dir/test_Node_$node" \
    --checkpoint_dir testing/test_Node_$node \
    --load_symbol_conformer checkpoints/AWGN_filter/SF8/best_conformer_jgj.pkl \
    --load checkpoints/denoiseGen/real_world/SF8 \
    --test_mode \
    --real_data \
    --free_gpu_id "$gpu_id"

  python $root_path_gloriphy/offset_correction.py \
    --data_dir "$real_data_dir/test_Node_$node" \
    --real_data
done

# Loop for NELoRa simulation tests
for sf in 8 9; do
  num_packets=100
  if [ "$sf" -eq 9 ]; then
    num_packets=120
  fi

  python $root_path_baseline/main_baseline.py \
    --root_path "$root_path_baseline" \
    --data_dir "$sim_data_dir/test_SF_${sf}_corrected" \
    --sf $sf \
    --num_packets $num_packets \
    --sim_data \
    --checkpoint_dir testing/test_SF${sf}_sim \
    --load checkpoints/simulation/SF${sf} \
    --test_mode \
    --free_gpu_id "$gpu_id"
done

# Loop for NELoRa real-world tests (Nodes D-H)
for node in "${nodes[@]}"; do
  python $root_path_baseline/main_baseline.py \
    --root_path "$root_path_baseline" \
    --data_dir "$real_data_dir/test_Node_${node}_corrected" \
    --real_data \
    --num_packets 200 \
    --checkpoint_dir testing/test_Node_$node \
    --load checkpoints/real-world/SF8 \
    --test_mode \
    --free_gpu_id "$gpu_id"
done

# for sf in 8 10 12; do # Uncomment for testing AWGN on SF8, SF10, and SF12
for sf in 8 10 ; do
  dir_path="$root_path_gloriphy/testing/awgn_tests_SF${sf}"
  mkdir -p "$dir_path"
  log_file_path="$dir_path/testing_log.csv"

  echo "Testing,SNR,Accuracy" > "$log_file_path"

  for snr in $(seq -35 2 -14); do
    transformer_dim=512
    if [ "$sf" -eq 8 ]; then
      transformer_dim=256
    fi

    python $root_path_gloriphy/main.py \
      --root_path "$root_path_gloriphy" \
      --train_denoiseGenCore \
      --sf $sf \
      --checkpoint_dir testing/awgn_tests_SF${sf} \
      --load checkpoints/AWGN_filter/SF${sf} \
      --test_mode \
      --free_gpu_id "$gpu_id" \
      --transformer_encoder_dim "$transformer_dim" \
      --transformer_layers 2 \
      --snr_list "$snr" 

    python $root_path_baseline/main_baseline.py \
      --root_path "$root_path_baseline" \
      --sf $sf \
      --checkpoint_dir testing/awgn_tests_SF${sf} \
      --load checkpoints/AWGN/SF${sf} \
      --test_mode \
      --test_awgn \
      --free_gpu_id "$gpu_id" \
      --snr_list "$snr" 

    python $root_path_gloriphy/check_LoRaPHY.py \
      --root_path "$root_path_gloriphy" \
      --sf $sf \
      --checkpoint_dir testing/awgn_tests_SF${sf} \
      --snr_list "$snr" 
  done
done

for transformer_dim in 256 1024; do
  dir_path="$root_path_gloriphy/testing/awgn_tests_SF10_emb${transformer_dim}"
  mkdir -p "$dir_path"
  log_file_path="$dir_path/testing_log.csv"
  echo "Testing,SNR,Accuracy" > "$log_file_path"

  for snr in $(seq -35 2 -14); do
    python $root_path_gloriphy/main.py \
      --root_path "$root_path_gloriphy" \
      --train_denoiseGenCore \
      --sf 10 \
      --checkpoint_dir "$dir_path" \
      --load "checkpoints/AWGN_filter/SF10/emb${transformer_dim}" \
      --test_mode \
      --free_gpu_id "$gpu_id" \
      --transformer_encoder_dim "$transformer_dim" \
      --transformer_layers 2 \
      --snr_list "$snr"    
  done
done


# #Plot results
python $root_path_gloriphy/plot_results.py
