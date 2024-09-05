import os
import json
import webdataset as wds
import librosa as lbs
import random
import soundfile as sf
import numpy as np
import pyrubberband as pyrb
from itertools import islice
import io
import multiprocessing
import pandas as pd
import copy
# import torchaudio


AUG_LIST = ['add_noise', 'bpm_shift', 'key_shift',None]
noise,sr = lbs.load('/data5/chaogang/mss/noise_48k.wav',sr=48000,mono=True)

# train_path_list0 = [
#     ["/data6/chaogang/downaudio/bad_quality_man_check1500_48k", 0],
#     ["/data6/chaogang/downaudio/audio_low_quality_295_check_id_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_rand3w_bad_3man_20240529_id_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_train_bad_need_check96_id_chaogang_check(含音质差字段)_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_version16_bad_0521_id_3man_20240521_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_version16_bad_prab0x9_7k_id_bad_0x8_bad_3man_20240523_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/dac1_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_疑似音质差-top10w片段-0.9门限-part5-20240424_id_chaogang_check_48k", 0],

#     ["/data5/chaogang/train_data_quality_orig/BabbleNoise_Driving_Machine_mix_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/noise_44kwav_30s_mp3_48k", 0],

#     ["/data5/chaogang/train_data_quality_orig/BabbleNoise_Driving_Machine_mix_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/noise_44kwav_30s_mp3_48k", 0],
# ]

# train_path_list = [
#     ["/data6/chaogang/downaudio/bad_quality_man_check1500_48k", 0],
#     ["/data6/chaogang/downaudio/audio_low_quality_295_check_id_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_rand3w_bad_3man_20240529_id_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_train_bad_need_check96_id_chaogang_check(含音质差字段)_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_version16_bad_0521_id_3man_20240521_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_version16_bad_prab0x9_7k_id_bad_0x8_bad_3man_20240523_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/dac1_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_疑似音质差-top10w片段-0.9门限-part5-20240424_id_chaogang_check_48k", 0],

#     ["/data5/chaogang/train_data_quality_orig/BabbleNoise_Driving_Machine_mix_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/noise_44kwav_30s_mp3_48k", 0],

#     ["/data5/chaogang/train_data_quality_orig/BabbleNoise_Driving_Machine_mix_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/noise_44kwav_30s_mp3_48k", 0],

#     ["/data5/chaogang/train_data_quality_orig/audio_0x_1x_700_48k", 1],
#     # ["/data6/chaogang/downaudio/audio_version1_top18k_48k", 1],
#     # ["/data6/chaogang/downaudio/audio_singer_SA_3w_48k", 1]
# ]
# valid_path_list0 = [
#     ["/data6/chaogang/downaudio/audio_bad116_id_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/noise_44kwav_30s_mp3_48k", 0]
# ]
# valid_path_list1 = [
#     ["/data6/chaogang/downaudio/audio_version1_top18k_48k_valid", 1],
#     ["/data6/chaogang/downaudio/audio_version1_top18k_48k", 1]
# ]
# valid_path_list = [
#     ["/data6/chaogang/downaudio/audio_bad116_id_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/noise_44kwav_30s_mp3_48k", 0],
    # ["/data6/chaogang/downaudio/audio_version1_top18k_48k_valid", 1],
# ]
# test_path_list = [
#     ["/data6/chaogang/downaudio/audio_bad116_id_44k", 0],
#     ["/data6/chaogang/downaudio/audio_version1_top18k_48k_valid", 1],
# ]

# tmp_train_list= [
#     ["/data6/chaogang/downaudio/bad_quality_man_check1500_48k", 0],
#     ["/data6/chaogang/downaudio/audio_low_quality_295_check_id_48k", 0],
#     ["/data5/chaogang/train_data_quality_orig/audio_0x_1x_700_48k", 1],
#     ["/data6/chaogang/downaudio/audio_version1_top18k_48k", 1]
# ]

# tmp_valid_list= [
#     ["/data6/chaogang/downaudio/audio_bad116_id_44k", 0],
#     ["/data6/chaogang/downaudio/audio_version1_top18k_48k_valid", 1],
# ]

def search_csv(path, target):
    df=pd.read_csv(path)
    return 'The music genre of this song is %s.' %df[df['hash_128'] == target]['genre']

def search_csv_multi_label(df, target):
    return list(df[df['0'] == target]['1'])

def get_all_files(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            files_list.append(os.path.join(root, file))
    return files_list

def add_real_noise(x, noise, snr=0):
    if len(noise)>len(x):
        start = random.randint(0, len(noise)-len(x))
        noise = noise[start:start+len(x)]
    else:
        start = random.randint(0,len(noise)-len(x)%len(noise))
        noise = ((len(x)//len(noise))*noise).append(noise[start:start+len(noise)-len(x)%len(noise)])
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = np.sum(noise ** 2) / len(noise)
    npower = xpower / (npower*snr + 1e-10)
    return x + noise*np.sqrt(npower)

def raw_dataset_samples_single_list(files, aug: str=None, audio_description: str=None, t: float=None, 
                                    # start: int=None, end: int=None
                                    ):
    #先拿到文件list再说好吧 彳亍
    # if start is None:
    #     start = 0
    # if end is None:
    #     end = len(files)
    # if end > len(files):
    #     end = len(files)
    # files = files[start:end]
    # samples = []
    df=pd.read_csv(audio_description)
    for fpath in files:
        if not (fpath.endswith(".wav") or fpath.endswith(".mp3")):
            continue
        
        duration = lbs.get_duration(path=fpath)
        if t != None:
            if duration<t:
                continue
            offset = random.uniform(0, duration-t)
            data, sr = lbs.load(fpath, sr=44100, offset=offset, duration=t, mono=False)
        else:
            try:
                data, sr = sf.read(fpath)
            except:
                print(fpath)
                continue
        # data = resampler(lbs.to_mono(data).float().cuda())
        data = lbs.resample(y=lbs.to_mono(data.T), orig_sr=sr, target_sr=22050, res_type='kaiser_fast')
        
        if aug=='add_noise':
            snr = random.uniform(-10, 10) # 信噪比
            data = add_real_noise(data, noise, snr)
        elif aug=='bpm_shift':
            r = random.uniform(0.8, 1.3) 
            data = pyrb.time_stretch(data, sr, r)
        elif aug=='key_shift':
            n_steps = random.randint(-6, 6)
            data = lbs.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

        # 创建io.BytesIO对象
        buffer = io.BytesIO()
        # 将音频数据保存为二进制数据
        sf.write(buffer, data, sr, format='WAV', subtype='PCM_16')
        # 获取二进制数据
        binary_data = buffer.getvalue()

        text = {"text": []}
        fname = fpath.split('/')[-1]
        if audio_description is not None:
            text['text'] = list(set(search_csv_multi_label(df, fname.split('.')[0])))

        # sf.write("tmp.wav", data, sr)
        # with open("tmp.wav", "rb") as stream:
        #     binary_data = stream.read()
        sample = {
            "__key__": os.path.splitext(fname)[0],
            "wav": binary_data,
            "json": json.dumps(text)
            }
        # samples.append(sample)
        yield sample

# with wds.ShardWriter("/data6/isaac/datasets/test/out-%06d.tar", maxcount=512) as sink:
#     for sample in islice(raw_dataset_samples(tmp_list), 0, 100):
#         sink.write(sample)
def write_wds(root, data_list, info, shared_dict, start_shard: int = 0,
            #    start_index: int = None, end_index: int=None,
                   aug: str=None, audio_dscrp: str=None, t: float=None):
    #create .tar file
    assert(info in ["train","valid","test"]), "Wrong Training Label!"
    assert(aug in AUG_LIST), "Wrong Data Augumentation Option!"
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(os.path.join(root, info)):
        os.makedirs(os.path.join(root, info))
    shard_num = 1+start_shard
    total_tmp1 = 0
    total_tmp2 = 1
    sizes_json = {}
    # if start_index == None:
    #     start_index=0
    #     max_num=start_index+max_num
    pid = str(os.getpid())
    while os.path.exists(os.path.join(root, info, pid+'-0.tar')):
        pid += 1
    # samples=raw_dataset_samples_single_list(data_list,aug,audio_dscrp,t)
    with wds.ShardWriter(pattern=os.path.join(root, info, pid+"-%d.tar"), maxcount=512, maxsize = 1e9, start_shard = start_shard) as sink:
    # with wds.ShardWriter(pattern=os.path.join(root, info, pid+"-%d.tar"), maxcount=512, maxsize = 1e9, start_shard = start_shard) as sink:
        for sample in islice(raw_dataset_samples_single_list(data_list,aug,audio_dscrp,t), 0, None):
            if shard_num==sink.shard:
                sink.write(sample)
                total_tmp1 = sink.total
            else: #if maximum conditions reached
                sizes_json[pid+'-'+str(shard_num-1)+".tar"]=sink.total-total_tmp2
                total_tmp2 = sink.total
                shard_num+=1
        sizes_json[pid+'-'+str(shard_num-1)+".tar"]=total_tmp1-total_tmp2+1
        ss = sink.shard
    #create sizes.json file
    # if os.path.exists(os.path.join(root, info, "sizes.json")):
    #     with open(os.path.join(root, info, "sizes.json"), "r") as f:
    #         data = json.load(f)
    #         sizes_json = {**data, **sizes_json}
    # with open(os.path.join(root, info, "sizes.json"), "w") as f:
    #     json.dump(sizes_json, f, indent = 4)
    for key in sizes_json.keys():
        shared_dict[key] = sizes_json[key]
    return ss


# def gen_aug_data(seed):
#     random.seed(seed)

#     save_path = "/data6/isaac/datasets/AudioQ_full_aug_" + str(seed)
#     pool = multiprocessing.Pool(processes=30)

#     pool.apply_async(write_wds, (save_path, train_path_list0, 'train', None,0,0,None,100))
#     # pool.apply_async(write_wds, (ii, fname, audioPath, pcmPath))

#     # process_args=[
#     #     {root, data_list, info, start_shard: int = 0, start_index: int = 0, max_num=None,  aug: str=None, audio_dscrp: str=None, t: float=None}
#     # ]

#     # for i, args in enumerate(process_args):

#     start_shard1 = write_wds(save_path, train_path_list0, "train", aug = 'add_noise',t=100)
#     start_shard2 = write_wds(save_path, [["/data5/chaogang/train_data_quality_orig/audio_0x_1x_700_48k", 1]], "train", start_shard=start_shard1, aug = 'bpm_shift',t=100)
#     start_shard3 = write_wds(save_path, [["/data5/chaogang/train_data_quality_orig/audio_0x_1x_700_48k", 1]], "train", start_shard=start_shard2, aug = 'key_shift',t=100)
#     start_shard4 =write_wds(save_path, [["/data6/chaogang/downaudio/audio_version1_top18k_48k", 1]], "train", limit=1500, start_shard=start_shard3, aug = 'bpm_shift',t=100)
#     start_shard5 =write_wds(save_path, [["/data6/chaogang/downaudio/audio_version1_top18k_48k", 1]], "train", limit=1500, start_shard=start_shard4, aug = 'key_shift',t=100)
#     start_shard6 = write_wds(save_path, [["/data6/chaogang/downaudio/audio_singer_SA_3w_48k", 1]], "train", limit=1500, start_shard=start_shard5, aug = 'bpm_shift',t=100)
#     s2 = write_wds(save_path, [["/data6/chaogang/downaudio/audio_singer_SA_3w_48k", 1]], "train", limit=1500, start_shard=start_shard6, aug = 'key_shift',t=100)

#     start_shard1 = write_wds(save_path, [['/data6/chaogang/downaudio/audio_bad1700_48k_100s', 0]], "train",start_shard=s2, aug = 'bpm_shift',limit=1400)
#     start_shard2 = write_wds(save_path, [['/data6/chaogang/downaudio/audio_bad1700_48k_100s', 0]], "train",start_shard=start_shard1, aug = 'key_shift',limit=1400)
#     start_shard3 = write_wds(save_path, [['/data6/chaogang/downaudio/audio_bad1700_48k_100s', 0]], "train",start_shard=start_shard2, aug = 'add_noise',limit=1400)
#     start_shard1 = write_wds(save_path, [['/data6/chaogang/downaudio/audio_multi_genres_1w4_48k_100s', 1]], "train",start_shard=start_shard3, aug = 'bpm_shift',limit=12000)
#     start_shard2 = write_wds(save_path, [['/data6/chaogang/downaudio/audio_multi_genres_1w4_48k_100s', 1]], "train",start_shard=start_shard1, aug = 'key_shift',limit=12000)


#     start_shard1 = write_wds(save_path, valid_path_list0, "valid", aug = 'add_noise',t=100)
#     start_shard2 = write_wds(save_path, valid_path_list1, "valid", limit=1500, aug = 'bpm_shift', start_shard=start_shard1, start_num=3000,t=100)
#     start_shard3 = write_wds(save_path, valid_path_list1, "valid", limit=1500, aug = 'key_shift', start_shard=start_shard2, start_num=3000,t=100)
#     start_shard1 = write_wds(save_path, [['/data6/chaogang/downaudio/audio_bad1700_48k_100s', 0]], "valid",start_shard=start_shard3, aug = 'bpm_shift', start_num=1401)
#     start_shard2 = write_wds(save_path, [['/data6/chaogang/downaudio/audio_bad1700_48k_100s', 0]], "valid",start_shard=start_shard1, aug = 'key_shift', start_num=1401)
#     start_shard3 = write_wds(save_path, [['/data6/chaogang/downaudio/audio_bad1700_48k_100s', 0]], "valid",start_shard=start_shard2, aug = 'add_noise', start_num=1401)
#     start_shard1 = write_wds(save_path, [['/data6/chaogang/downaudio/audio_multi_genres_1w4_48k_100s', 1]], "valid",start_shard=start_shard3, aug = 'bpm_shift', start_num=12001)
#     start_shard2 = write_wds(save_path, [['/data6/chaogang/downaudio/audio_multi_genres_1w4_48k_100s', 1]],"valid",start_shard=start_shard1, aug = 'key_shift', start_num=12001)



if __name__ == "__main__":
    save_path = "/data6/isaac/datasets/song_label"
    cpu_num = 35

    process_args=[]
    train_list = get_all_files('/data6/chaogang/song_label/audio_label_train')
    valid_list = get_all_files('/data6/chaogang/song_label/audio_label_valid')
    test_list = get_all_files('/data6/chaogang/song_label/audio_label100_test')
    train_cpu = int(len(train_list)/(len(train_list)+len(valid_list)+len(test_list))*cpu_num)
    valid_cpu = int(len(valid_list)/(len(train_list)+len(valid_list)+len(test_list))*cpu_num)
    test_cpu = cpu_num-train_cpu-valid_cpu

    cpu_num = train_cpu + valid_cpu + test_cpu
    pool = multiprocessing.Pool(processes=cpu_num+3)
    manager = multiprocessing.Manager()
    shared_dict = {"train": manager.dict(), "valid": manager.dict(), "test": manager.dict()}

    r = list(range(0,len(train_list)-len(train_list)%train_cpu,len(train_list)//train_cpu))+[len(train_list)]
    r1 = copy.deepcopy(r);r1.pop()
    r2 = list(r[1:])
    tmp = len(r2)
    # r = range(0,158229,6000)
    # r1 = list(r);r1.pop()
    # r2 = list(r[1:])
    for start, end in zip(r1, r2):
        process_args.append(
            {
                'root':save_path, 
                'data_list':train_list[start:end], 
                'info':'train', 
                'shared_dict': shared_dict["train"],
                'start_shard' : 0,
                # 'start_index' : start,
                # 'end_index': end,  
                'aug':None, 
                'audio_dscrp':'/data6/chaogang/song_label/hash_ipname_res_ziwen_train2.csv', 
                't':None
            }
        )

    r = list(range(0,len(valid_list)-len(valid_list)%valid_cpu,len(valid_list)//valid_cpu))+[len(valid_list)]
    r1 = list(r);r1.pop()
    r2 = list(r[1:])
    tmp += len(r2)
    # r = range(0,51581,6000)
    # r1 = list(r);r1.pop()
    # r2 = list(r[1:])
    for start, end in zip(r1, r2):
        process_args.append(
            {
                'root':save_path, 
                'data_list':valid_list[start:end], 
                'info':'valid', 
                'shared_dict': shared_dict["valid"],
                'start_shard' : 0,
                # 'start_index' : start,
                # 'end_index': end,  
                'aug':None, 
                'audio_dscrp':'/data6/chaogang/song_label/hash_ipname_res_ziwen_valid2.csv', 
                't':None
            }
        )
    
    r = list(range(0,len(test_list)-len(test_list)%test_cpu,len(test_list)//test_cpu))+[len(test_list)]
    r1 = list(r);r1.pop()
    r2 = list(r[1:])
    tmp += len(r2)
    for start, end in zip(r1, r2):
        process_args.append(
            {
                'root':save_path, 
                'data_list':test_list[start:end], 
                'info':'test',
                'shared_dict': shared_dict["test"],
                'start_shard': 0,
                # 'start_index' : start,
                # 'end_index': end,  
                'aug':None, 
                'audio_dscrp':'/data6/chaogang/song_label/song_label_test100.csv', 
                't':None
            }
        )

    for i, args in enumerate(process_args):
        pool.apply_async(write_wds, tuple(args.values()))
    pool.close()
    pool.join()

    for info in ["train","valid","test"]:
        with open(os.path.join(save_path, info, "sizes.json"), "w") as f:
            json.dump(dict(shared_dict[info]), f, indent = 4)

