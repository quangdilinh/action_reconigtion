#!/bin/bash
# Set the path to save checkpoints and specify the new directory format
OUTPUT_DIR='pickles/A2'
DATA_PATH='data/A2/'
BATCH_SIZE=6
NUM_WORKER=6
NUM_SAMPLE=1
NUM_FRAMES=16
WEIGHT_DECAY=0.05
LEARNING_RATE=2e-3
MODEL_PATH_PREFIX='./checkpoints'

# Define an array of parameter names and values
SAMPLING_RATES=(15 10 20)
CLIP_STRIDES=(30 20 15 10 5)

# Loop through parameter combinations
for clip_stride in "${CLIP_STRIDES[@]}"; do
  for sampling_rate in "${SAMPLING_RATES[@]}"; do
    # Create a new directory based on the specified format
    NEW_DIR="pickles/${BATCH_SIZE}_${NUM_WORKER}_${NUM_SAMPLE}_${NUM_FRAMES}_${sampling_rate}_${WEIGHT_DECAY}_${LEARNING_RATE}_${clip_stride}"
    mkdir -p "$NEW_DIR"

    # Create a text file with the input parameters
    echo "BATCH_SIZE=$BATCH_SIZE" > "$NEW_DIR/params.txt"
    echo "NUM_WORKER=$NUM_WORKER" >> "$NEW_DIR/params.txt"
    echo "NUM_SAMPLE=$NUM_SAMPLE" >> "$NEW_DIR/params.txt"
    echo "NUM_FRAMES=$NUM_FRAMES" >> "$NEW_DIR/params.txt"
    echo "SAMPLING_RATE=$sampling_rate" >> "$NEW_DIR/params.txt"
    echo "WEIGHT_DECAY=$WEIGHT_DECAY" >> "$NEW_DIR/params.txt"
    echo "LEARNING_RATE=$LEARNING_RATE" >> "$NEW_DIR/params.txt"
    echo "CLIP_STRIDE=$clip_stride" >> "$NEW_DIR/params.txt"

    # Loop through different views and folds
    views=("dash" "right" "rear")
    folds=(0 1 2 3 4)

    for view in "${views[@]}"; do
      for fold in "${folds[@]}"; do
        MODEL_PATH="${MODEL_PATH_PREFIX}/${view}_${fold}.pth"

        # Use a single CUDA_VISIBLE_DEVICES and a single evaluate_loc.py call
        CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
          --model vit_large_patch16_224 \
          --data_set Kinetics-400 \
          --view "$view" \
          --fold $fold \
          --nb_classes 16 \
          --data_path $DATA_PATH \
          --finetune $MODEL_PATH \
          --log_dir $OUTPUT_DIR \
          --output_dir $OUTPUT_DIR \
          --batch_size $BATCH_SIZE \
          --num_worker $NUM_WORKER \
          --num_sample $NUM_SAMPLE \
          --input_size 224 \
          --short_side_size 224 \
          --save_ckpt_freq 10 \
          --num_frames $NUM_FRAMES \
          --sampling_rate $sampling_rate \
          --opt lion \
          --opt_betas 0.9 0.99 \
          --weight_decay $WEIGHT_DECAY \
          --epochs 35 \
          --lr $LEARNING_RATE \
          --clip_stride $clip_stride \
          --crop
      done
    done

    # Copy files that match the pattern to the new directory
    cp "$OUTPUT_DIR/A1*.pkl" "$NEW_DIR/"

    # Run the submission script
    python run_submission.py

    # Copy A2_submission.txt to the new directory
    cp "$OUTPUT_DIR/A2_submission.txt" "$NEW_DIR/"


  done
done
