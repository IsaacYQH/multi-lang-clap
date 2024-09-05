import os
import torch
import librosa
import numpy as np
import pprint
import pandas as pd
import random
random.seed(0)
from itertools import islice
from tqdm import tqdm
import datetime

import os
import sys
 
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(BASE_DIR)

from clap_module import create_model
from training.data import get_audio_features
from training.data import int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer
from transformers import XLMRobertaTokenizer

import torch.nn.functional as F

def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    # distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    distmat = torch.mm(input1_normed, input2_normed.t())
    return distmat

tokenize = XLMRobertaTokenizer.from_pretrained("/data2/isaac/cofac/myclap/checkpoints/xlm-roberta-base")
def tokenizer(text):
    result = tokenize(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in result.items()}

def infer_text(text_data, model, model_cfg):
    # load the text, can be a list (i.e. batch size)
    # text_data = ["I love the contrastive learning", "I love the pretrain model"] 
    # tokenize for roberta, if you want to tokenize for another text encoder, please refer to data.py#L43-90 
    text_data = tokenizer(text_data)
    model.eval()
    text_embed = model.get_text_embedding(text_data)
    text_embed = text_embed.detach().cpu().numpy()
    # print(text_embed)
    # print(text_embed.shape)
    return text_embed

def infer_audio(path, model, model_cfg):
    # load the waveform of the shape (T,), should resample to 48000
    audio_waveform, sr = librosa.load(path, sr=48000, mono=False)
    if audio_waveform.shape[0]==2:
        if random.uniform(0, 1) > 0.5:
            audio_waveform = audio_waveform[0, :]
        else:
            audio_waveform = audio_waveform[1, :]
    # quantize
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
    audio_waveform = torch.from_numpy(audio_waveform).float()
    audio_dict = {}

    # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
    # audio_dict = get_audio_features(
    #     audio_dict, audio_waveform, 480000, 
    #     data_truncating='fusion', 
    #     data_filling='repeatpad',
    #     audio_cfg=model_cfg['audio_cfg']
    # )
    # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
    audio_dict = get_audio_features(
        audio_dict, audio_waveform, 480000, 
        data_truncating='rand_trunc', 
        data_filling='repeatpad',
        audio_cfg=model_cfg['audio_cfg']
    )
    model.eval()
    # can send a list to the model, to process many audio tracks in one time (i.e. batch size)
    audio_embed = model.get_audio_embedding([audio_dict])
    audio_embed = audio_embed.detach().cpu().numpy()
    # print(audio_embed)
    # print(audio_embed.shape)
    return audio_embed

def get_model(pretrained, amodel, tmodel, precision: str='fp32',enable_fusion: bool=False, fusion_type: str='aff_2d'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if enable_fusion:
        model, model_cfg = create_model(
            amodel,
            tmodel,
            pretrained,
            precision=precision,
            device=device,
            enable_fusion=enable_fusion,
            fusion_type=fusion_type
        )
    else:
        model, model_cfg = create_model(
            amodel,
            tmodel,
            pretrained,
            precision=precision,
            device=device,
            enable_fusion=enable_fusion
        )
        # model = laion_clap.CLAP_Module(
        #     enable_fusion=False,
        #     device=device,
        #     amodel= 'HTSAT-base',
        #     tmodel='roberta'
        #    # fusion_type=fusion_type
        #     )
        # model.load_ckpt(ckpt=pretrained)
    return model, model_cfg

    
def get_file_list(audiopath):
    filelist = []
    for ifile in os.listdir(audiopath):
        if ".wav" in ifile or ".mp3" in ifile:
            file1 = os.path.join(audiopath, ifile)
            filelist.append(file1)
    return filelist

def write_csv(model, model_cfg, file_name=str, bad_lists: list=None, good_lists: list=None, bad_start: int=None, bad_end: int=None, good_start: int=None, good_end: int=None):
    results_from_good = {'wrong': 0, 'right': 0}
    results_from_bad = {'wrong': 0, 'right': 0}

    df = pd.DataFrame({'path' : [], 'label' : [], 'dist_good' : [], 'dist_bad' : [], 'dist_dif_g-b' : []})
    if bad_lists!=None:
        for i in bad_lists:
            list = get_file_list(i)[bad_start:bad_end]
            count=0
            for item in list:
                tmp = torch.from_numpy(infer_audio(item,model, model_cfg))
                dist_from_good = cosine_distance(tmp,torch.from_numpy(good_text_emb).unsqueeze(0)).numpy()
                dist_from_bad = cosine_distance(tmp,torch.from_numpy(bad_text_emb).unsqueeze(0)).numpy()
                df=df._append({'path' : item,'label' : '0','dist_good' :dist_from_good[0][0],'dist_bad' :dist_from_bad[0][0], 'dist_dif_g-b' :dist_from_good[0][0]-dist_from_bad[0][0]},ignore_index = True)
                if dist_from_good < dist_from_bad:
                    results_from_bad['right']+=1
                else:
                    results_from_bad['wrong']+=1
                count+=1
                print(count)

    if good_lists!=None:
        for i in good_lists:
            list = get_file_list(i)[good_start:good_end]
            count=0
            for item in list:
                tmp = torch.from_numpy(infer_audio(item,model, model_cfg))
                dist_from_good = cosine_distance(tmp,torch.from_numpy(good_text_emb).unsqueeze(0)).numpy()
                dist_from_bad = cosine_distance(tmp,torch.from_numpy(bad_text_emb).unsqueeze(0)).numpy()
                df=df._append({'path' : item,'label' : '1','dist_good' :dist_from_good[0][0],'dist_bad' :dist_from_bad[0][0], 'dist_dif_g-b' :dist_from_good[0][0]-dist_from_bad[0][0]},ignore_index = True)
                if dist_from_good > dist_from_bad:
                    results_from_good['right']+=1
                else:
                    results_from_good['wrong']+=1
                count+=1
                print(count)

    df.to_csv('results/'+ file_name + '.csv', index = False)
    if (results_from_good['right']+results_from_good['wrong'])!=0:
        print('good: %f' % (results_from_good['right']/(results_from_good['right']+results_from_good['wrong'])))
    if (results_from_bad['right']+results_from_bad['wrong'])!=0:
        print('bad: %f' % (results_from_bad['right']/(results_from_bad['right']+results_from_bad['wrong'])))

def write_csv_genre(model, model_cfg, root, genre_chart, genre_text_emb=dict, file_name=str):
    # target_dataframe = pd.read_csv(target,encoding=encoding)
    df = pd.DataFrame({'path' : [], 'label' : []}.update({k: [] for k in genre_text_emb.keys()}))
    for i, [item, label] in tqdm(genre_chart.iterrows(), desc='Processing'):
        tmp = torch.from_numpy(infer_audio(os.path.join(root,item+'.wav'), model, model_cfg))
        dist_dict = {}
        for key, value in genre_text_emb.items():
            dist_dict.update({str('dist_from_'+key): cosine_distance(tmp,torch.from_numpy(value).unsqueeze(0)).numpy()[0][0]})
        dist_values_list = list(dist_dict.values())
        dist_key_list = list(dist_dict.keys())
        sorted_list = sorted(dist_values_list, reverse=True)
        df=df._append({**{'path' : item, 'label' : label, 
                       'top1_label': dist_key_list[dist_values_list.index(sorted_list[0])].replace('dist_from_',''),
                       'top2_label': dist_key_list[dist_values_list.index(sorted_list[1])].replace('dist_from_',''),
                       'top3_label': dist_key_list[dist_values_list.index(sorted_list[2])].replace('dist_from_',''),
                       'top4_label': dist_key_list[dist_values_list.index(sorted_list[3])].replace('dist_from_',''),
                       'top5_label': dist_key_list[dist_values_list.index(sorted_list[4])].replace('dist_from_',''),},
                    #    **dist_dict
                       },ignore_index = True)
    if not os.path.exists('results'):
        os.mkdir('results')
    try:
        df.to_csv('results/'+ file_name + '.csv', index = False)
    except:
        df.to_csv('~/file_name.csv', index = False)
        print("csv file stored in home dir.")


if __name__ == "__main__":
    checkpoint = '/data2/isaac/cofac/myclap/checkpoints/epoch_latest.pt'
    map_path='/data2/chaogang/data_song_label/audio_label100_test.xlsx'
    model, model_cfg = get_model(
    amodel = 'HTSAT-base', 
    tmodel = 'xlm-roberta',
    pretrained = checkpoint
    )
    
    # 检测文件的编码格式
    # with open('/data2/chaogang/data_song_label/audio_label100_test.xlsx', 'rb') as f:
    #     result = chardet.detect(f.read())
    # encoding = result['encoding']
    genre_chart = pd.read_excel(map_path,)[['hash_128','TAG名']]
    genre_list = list(genre_chart['TAG名'].unique())
    genre_emb = infer_text(
        genre_list,
        model,
        model_cfg
        )
    genre_dict = {k: v for k, v in zip(genre_list, genre_emb)}

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch = os.path.basename(checkpoint).split('.')[0]
    root  ='/data2/chaogang/data_song_label/audio_label100_test_22k'
    write_csv_genre(model, model_cfg, root, genre_chart, genre_dict, 
                    file_name=f'{current_time}_{os.path.basename(map_path).split(".")[0]}_{epoch}')
