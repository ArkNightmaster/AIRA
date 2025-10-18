#!/bin/bash

# This script runs training and evaluation for AIRA_MoE, Vera, HydraLora, and LoRA
# on the Law dataset in parallel on 8 GPUs.

# Create log directory
mkdir -p ./output/logs

module add cuda/12.4

# --- AIRA_MoE on GPU 0,1 ---
echo "Starting AIRA_MoE training and evaluation on GPUs 0,1..."
(
  export CUDA_VISIBLE_DEVICES=0,1
  bash ./scripts/aira_moe/law.sh
) > ./output/logs/aira_moe_law.log 2>&1 &
AIRA_PID=$!
echo "AIRA_MoE process started with PID: $AIRA_PID"

# # --- Vera on GPU 2,3 ---
# echo "Starting Vera training and evaluation on GPUs 2,3..."
# (
#   export CUDA_VISIBLE_DEVICES=2,3
#   bash ./scripts/cola/law.sh
# ) > ./output/logs/cola_law.log 2>&1 &
# VERA_PID=$!
# echo "Cola process started with PID: $VERA_PID"

# # --- HydraLora on GPU 4,5 ---
# echo "Starting HydraLora training and evaluation on GPUs 4,5..."
# (
#   export CUDA_VISIBLE_DEVICES=4,5
#   bash ./scripts/hydralora/law.sh
# ) > ./output/logs/hydralora_law.log 2>&1 &
# HYDRA_PID=$!
# echo "HydraLora process started with PID: $HYDRA_PID"

# # --- LoRA on GPU 6,7 ---
# echo "Starting LoRA training and evaluation on GPUs 6,7..."
# (
#   export CUDA_VISIBLE_DEVICES=6,7
#   bash ./scripts/lora/law.sh
# ) > ./output/logs/lora_law.log 2>&1 &
# LORA_PID=$!
# echo "LoRA process started with PID: $LORA_PID"

# Wait for all background jobs to finish
echo "Waiting for all processes to complete..."
wait $AIRA_PID
echo "AIRA_MoE finished."
wait $VERA_PID
echo "Vera finished."
wait $HYDRA_PID
echo "HydraLora finished."
wait $LORA_PID
echo "LoRA finished."

echo "All jobs completed. Logs are in ./output/logs"
