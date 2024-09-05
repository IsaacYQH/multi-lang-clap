PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=gpu  -m training.main\
    --save-frequency 20 \ #保存checkpoints的频率
    --save-most-recent \ #保存最近一个epoch
    --dataset-type="webdataset" \
    --datasetpath="/home/isaac/datasets" \#数据集的根目录
    --precision="fp32" \
    --batch-size=64 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=30 \
    --workers=6 \
    --use-bn-sync \
    --amodel HTSAT-base \ #audio encoder
    --tmodel xlm-roberta \ #text encoder
    --warmup 0 \
    --datasetnames "nsynth_web" \ #数据集名称
    --report-to "tensorboard" \
    --datasetinfos "train" \
    --save-top-performance 3 \ #根据test set保存top几的checkpoints
    --top-k-checkpoint-select-dataset="nsynth_web-test"\ #测试集名称，必须要在根目录里面有才可以写名字
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs_dir 'logs' \ #logs路径
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --prefetch-factor 2\
    --pretrained /home/isaac/checkpoints/clap-ckpt.pt #预训练的checkpoint
    # --resume /data6/isaac/cofac/CLAP_orig/src/laion_clap/logs/2024_07_11-09_47_13-model_HTSAT-base-lr_0.0001-b_64-j_6-p_fp32/checkpoints/epoch_latest.pt #从哪个checkpoints继续

# debug code
# python -m debugpy --wait-for-client --listen 2222 training/main.py\
#     --save-frequency 5 \
#     --save-top-performance 3 \
#     --save-most-recent \
#     --dataset-type="webdataset" \
#     --datasetpath="/data6/isaac/datasets" \
#     --precision="fp32" \
#     --batch-size=64 \
#     --lr=1e-4 \
#     --wd=0.0 \
#     --epochs=100 \
#     --workers=6 \ 
#     --use-bn-sync \
#     --amodel HTSAT-base \
#     --tmodel xlm-roberta \
#     --warmup 3200 \
#     --datasetnames "song_label" \
#     --report-to "tensorboard" \
#     --datasetinfos "train" \
#     --top-k-checkpoint-select-dataset="song_label-test"\
#     --top-k-checkpoint-select-metric="mAP@10" \
#     --logs 'logs' \
#     --seed 3407 \
#     --gather-with-grad \
#     --optimizer "adam" \
#     --data-filling "repeatpad" \
#     --data-truncating "rand_trunc" \
#     # --pretrained /data6/isaac/music_audioset_epoch_15_esc_90.14.pt\
#     --pretrained-audio /data5/chaogang/project/CLAP-main/HTSAT_AudioSet_Saved_music_audioset_epoch_15_esc_90.14.pt \
#     --prefetch-factor 2 \
#     # --resume /data6/isaac/cofac/CLAP_orig/src/laion_clap/logs/2024_07_11-09_47_13-model_HTSAT-base-lr_0.0001-b_64-j_6-p_fp32/checkpoints/epoch_latest.pt

