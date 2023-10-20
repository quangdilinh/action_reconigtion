# Set the path to save checkpoints
OUTPUT_DIR='pickles/A2'
DATA_PATH='data/A2/'
BATCH_SIZE=6
NUM_WORKER=6
NUM_SAMPLE=1
NUM_FRAMES=16
# fine tune this
SAMPLING_RATE=15
WEIGHT_DECAY=0.05
LEARNING_RATE=2e-3
# fine tune this
CLIP_STRIDE=30

MODEL_PATH='./checkpoints/dash_0.pth'
# batch_size can be adjusted according to the number of GPUs
# This script is for 64 GPUs (8 nodes x 8 GPUs)
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "dash" \
    --fold 0 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \

MODEL_PATH='./checkpoints/dash_1.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "dash" \
    --fold 1 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \



MODEL_PATH='./checkpoints/dash_2.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "dash" \
    --fold 2 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \


MODEL_PATH='./checkpoints/dash_3.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "dash" \
    --fold 3 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \

MODEL_PATH='./checkpoints/dash_4.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "dash" \
    --fold 4 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \

MODEL_PATH='./checkpoints/rightside_0.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "right" \
    --fold 0 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \

MODEL_PATH='./checkpoints/rightside_1.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "right" \
    --fold 1 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \

MODEL_PATH='./checkpoints/rightside_2.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "right" \
    --fold 2 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \

MODEL_PATH='./checkpoints/rightside_3.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "right" \
    --fold 3 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \

MODEL_PATH='./checkpoints/rightside_4.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "right" \
    --fold 4 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \

MODEL_PATH='./checkpoints/rearview_0.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "rear" \
    --fold 0 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \


MODEL_PATH='./checkpoints/rearview_1.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --fold 1 \
    --view "rear" \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \


MODEL_PATH='./checkpoints/rearview_2.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "rear" \
    --fold 2 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \



MODEL_PATH='./checkpoints/rearview_3.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "rear" \
    --fold 3 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \



MODEL_PATH='./checkpoints/rearview_4.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --view "rear" \
    --fold 4 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_worker ${NUM_WORKER} \
    --num_sample ${NUM_SAMPLE} \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs 35 \
    --lr ${LEARNING_RATE} \
    --clip_stride ${CLIP_STRIDE} \
    --crop \