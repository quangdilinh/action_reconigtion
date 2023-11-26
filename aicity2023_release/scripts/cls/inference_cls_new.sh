#!/bin/bash
# Set the script to exit on error and unset variable

# Set the path to save checkpoints
OUTPUT_DIR='pickles/A2'
DATA_PATH='data/A2/'
BATCH_SIZE=12
NUM_WORKER=16
NUM_SAMPLE=1
NUM_FRAMES=16
WEIGHT_DECAY=0.05
LEARNING_RATE=0.002

# Define an array of sampling rates
SAMPLING_RATES=(4)

# Define an array of clip strides
CLIP_STRIDES=(30)

# Define the views and folds
views=("dash" "right" "rear")
folds=(0 1 2 3 4)

# Function to create a new directory and set the checkpoint values
create_new_directory() {
  local NEW_DIR="$1"
  mkdir -p "$NEW_DIR"
}

# Function to run model evaluation
run_model_evaluation() {
  local view="$1"
  local fold="$2"
  local clip_stride="$3"
  local sampling_rate="$4"
  local NEW_DIR="$5"

  MODEL_PATH="./checkpoints/${view}_${fold}.pth"
  CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "$view" \
    --fold "$fold" \
    --nb_classes 16 \
    --data_path "${DATA_PATH}" \
    --finetune "$MODEL_PATH" \
    --log_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_worker "$NUM_WORKER" \
    --num_sample "$NUM_SAMPLE" \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames "$NUM_FRAMES" \
    --sampling_rate "$sampling_rate" \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay "$WEIGHT_DECAY" \
    --epochs 35 \
    --lr "$LEARNING_RATE" \
    --clip_stride "$clip_stride" \
    --crop
  
  if [[ $? -eq 0 ]]; then
    # Copy the weight file to NEW_DIR
    cp "$OUTPUT_DIR/A1_${view}_vmae_16x4_crop_fold_${fold}.pkl" "$NEW_DIR/"
    echo "Evaluation succeeded"
    return 0
  else
    # Handle the error condition here
    echo "Evaluation failed"
    return 1
  fi
  
}


run_model_evaluation_with_retry() {
  local view="$1"
  local fold="$2"
  local clip_stride="$3"
  local sampling_rate="$4"
  local NEW_DIR="$5"
  local max_retries=3
  local retry_count=0
  local gap_btw_retry=5

  while [ $retry_count -lt $max_retries ]; do
    echo "Running evaluation for view: $view, fold: $fold, clip_stride: $clip_stride, sampling_rate: $sampling_rate (Attempt $((retry_count+1)))"
    run_model_evaluation "$view" "$fold" "$clip_stride" "$sampling_rate" "$NEW_DIR"

    if [ $? -eq 0 ]; then
      # Evaluation successful, break out of the retry loop
      break
    else
      # Evaluation failed, increment the retry count and sleep for a while before retrying
      retry_count=$((retry_count+1))
      echo "Sleeping for $gap_btw_retry s"
      sleep $gap_btw_retry  # You can adjust the sleep duration as needed

    fi
  done

  # Check if the maximum number of retries has been reached
  if [ $retry_count -eq $max_retries ]; then
    echo "Max retries reached for view: $view, fold: $fold, clip_stride: $clip_stride, sampling_rate: $sampling_rate. Ending the script."
    exit 1
  fi
}


# Function to create a params.txt file with input parameters
create_params_file() {
  local NEW_DIR="$1"
  echo "BATCH_SIZE=$BATCH_SIZE" > "$NEW_DIR/params.txt"
  echo "NUM_WORKER=$NUM_WORKER" >> "$NEW_DIR/params.txt"
  echo "NUM_SAMPLE=$NUM_SAMPLE" >> "$NEW_DIR/params.txt"
  echo "NUM_FRAMES=$NUM_FRAMES" >> "$NEW_DIR/params.txt"
  echo "SAMPLING_RATE=$sampling_rate" >> "$NEW_DIR/params.txt"
  echo "WEIGHT_DECAY=$WEIGHT_DECAY" >> "$NEW_DIR/params.txt"
  echo "LEARNING_RATE=$LEARNING_RATE" >> "$NEW_DIR/params.txt"
  echo "CLIP_STRIDE=$clip_stride" >> "$NEW_DIR/params.txt"
  echo "BATCH_SIZE=$BATCH_SIZE NUM_WORKER=$NUM_WORKER NUM_SAMPLE=$NUM_SAMPLE NUM_FRAMES=$NUM_FRAMES"
  echo "SAMPLING_RATE=$sampling_rate WEIGHT_DECAY=$WEIGHT_DECAY LEARNING_RATE=$LEARNING_RATE CLIP_STRIDE=$clip_stride"
  }



# Central checkpoint file
CHECKPOINT_FILE="pickles/inference_checkpoint.txt"

# Check if a central inference checkpoint file exists and resume from it
if [ -f "$CHECKPOINT_FILE" ]; then
  read -r checkpoint_clip_stride checkpoint_sampling_rate checkpoint_view checkpoint_fold < "$CHECKPOINT_FILE"
  echo "Resuming from checkpoint: clip_stride=$checkpoint_clip_stride, sampling_rate=$checkpoint_sampling_rate, view=$checkpoint_view, fold=$checkpoint_fold"
  found_checkpoint=false
else
  found_checkpoint=true
fi

# Main loop to iterate through parameter combinations
for clip_stride in "${CLIP_STRIDES[@]}"; do
  for sampling_rate in "${SAMPLING_RATES[@]}"; do
    NEW_DIR="pickles/${BATCH_SIZE}_${NUM_WORKER}_${NUM_SAMPLE}_${NUM_FRAMES}_${sampling_rate}_${WEIGHT_DECAY}_${LEARNING_RATE}_${clip_stride}"
    if [ "$found_checkpoint" = true ]; then
      create_new_directory "$NEW_DIR"
      create_params_file "$NEW_DIR"
    fi
    
    for view in "${views[@]}"; do
      for fold in "${folds[@]}"; do
        # Skip until the checkpoint values are reached
        if [ "$found_checkpoint" = false ]; then
          if [ "$clip_stride" -eq "$checkpoint_clip_stride" ] && [ "$sampling_rate" -eq "$checkpoint_sampling_rate" ] && [ "$view" = "$checkpoint_view" ] && [ "$fold" -eq "$checkpoint_fold" ]; then
            found_checkpoint=true
          fi
        fi

        
        if [ "$found_checkpoint" = true ]; then
          # Update the central checkpoint file
          echo "$clip_stride $sampling_rate $view $fold" > "$CHECKPOINT_FILE"

          # Get the current parameter values
          
          run_model_evaluation_with_retry "$view" "$fold" "$clip_stride" "$sampling_rate" "$NEW_DIR"
          
          # Update the central checkpoint file
          echo "$clip_stride $sampling_rate $view $fold" > "$CHECKPOINT_FILE"
        else
          echo "Skipping: clip_stride=$clip_stride, sampling_rate=$sampling_rate, view=$view, fold=$fold"
        fi
      done
    done
    if [ "$found_checkpoint" = true ]; then
      # clear checkpoint after finished param
      rm "$CHECKPOINT_FILE"
      python run_submission.py
      cp "A2_submission.txt" "$NEW_DIR/"
    fi
  done
done
