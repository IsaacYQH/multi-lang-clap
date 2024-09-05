## Copy important files and dirs

```bash
/data5/isaac/xlm-roberta-base
/data6/isaac/datasets/song_label
/data6/isaac/datasets/multi_genres_1w4_genList.csv
/data6/isaac/cofac/myclap/logs/2024_08_13-16_41_48-model_HTSAT-base-lr_0.0001-b_80-j_6-p_fp32/
```

## Environment Setup
```bash
pip3 install -r requirements.txt
```

## Prepare webdataset
```bash
python utils\gen_webdataset.py
```

## Train
```bash
bash train_script.sh
```
## Test
```bash
python test\infer_Genre_demo.py
```
