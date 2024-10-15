## Copy important files and dirs
If your server is offline, please download files needed for the following model loading.
```python
RobertaTokenizer.from_pretrained("roberta-base")
```

## Environment Setup
```bash
pip3 install -r requirements.txt
```

## Prepare webdataset
For example nsynth:
```bash
python utils\data_rearrange_nsynth.py
```
Or convert from lmdb to webdataset
```bash
python utils\lmdb_to_webdataset.py
```

## Train
```bash
bash train_script.sh
```
Related parameters used in training is explained in the shell script. If more information is needed, refer to ‘params.py’

