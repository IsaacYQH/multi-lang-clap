'''
Params:
    save-frequency (n: int): checkpoint saved every n epoch
    save-most-recent (boolean): save the most recent checkpoint or not
    dataset-type (str)
    datasetpath (str)
    precision (str)
    batch-size (int)
    lr (float): learning rate
    wd (float): weight decay
    epochs (int)
    workers (int)
    use-bn-sync (boolean): Whether to use batch norm sync.
    amodel (str): audio model
    tmodel (str): text model
    warmup (int): warm up steps
    datasetnames (multiple str)
    report-to (str): wandb or tensorboard
    datasetinfos (str): 'train'/'valid'/'test'
    save-top-performance (n: boolean): save top n performance checkpoints based on test set
    top-k-checkpoint-select-dataset (str): save top n performance checkpoints based on test set of which dataset (naming template: 'datasetname-test')
    top-k-checkpoint-select-metric (str)
    logs_dir (str): log dir path
    seed (int)
    gather-with-grad (boolean): enable full distributed gradient for feature gather
    optimizer (str)
    data-filling (str): data filling method including repeat, repeatpad, pad
    data-truncating (str): data truncating method including rand_trunc, fusion
    prefetch-factor (int): The prefetch factor for dataloader. Larger value will use more memory and CPU but faster.
    pretrained (str): path of prtrained clap checkpoints
    resume (str): path of previous training checkpoints if resuming training is needed.
'''

PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=gpu  -m training.main\
    --save-frequency 5 \
    --save-most-recent\
    --dataset-type="webdataset"\
    --datasetpath="/lan/datasetsAlpha4090"\
    --precision="fp32"\
    --batch-size=64\
    --lr=1e-4\
    --wd=0.0\
    --epochs=30\
    --workers=240\
    --use-bn-sync\
    --amodel HTSAT-base\
    --tmodel roberta\
    --warmup 500\
    --datasetnames "nsynth_webdataset" "urmp_webdataset" "slakh2100_webdataset"\
    --report-to "wandb"\
    --datasetinfos "train"\
    --save-top-performance 3\
    --top-k-checkpoint-select-dataset="nsynth_webdataset-test"\
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs_dir 'logs' \
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "fusion" \
    --prefetch-factor 2\
    --pretrained /home/isaac/checkpoints/clap-ckpt.pt
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

