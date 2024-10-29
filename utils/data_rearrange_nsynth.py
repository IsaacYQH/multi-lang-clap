import os
import json
import webdataset as wds
import soundfile as sf
from itertools import islice
import io
import multiprocessing
import copy
# TODO: change generation of json, generate after tar

NOTE_QUA_LIST = [
    "Bright, a bright tone or brightness means a large amount of high frequency content and strong upper harmonics.",
    "Dark, a dark tone or darkness means a distinct lack of high frequency content, giving a muted and bassy sound. Also sometimes described as Warm.",
    "Distortion means waveshaping that produces a distinctive crunchy sound and presence of many harmonics. Sometimes paired with non-harmonic noise.",
    "Fast decay means amplitude envelope of all harmonics decays substantially before the ‘note-off’ point or offset of each note at 3 seconds.",
    "Long release means amplitude envelope decays slowly after the ‘note-off’ point or offset of each note, sometimes still present at the end of the sample 4 seconds.",
    "Multiphonic means 	Presence of overtone frequencies related to more than one fundamental frequency.",
    "Nonlinear env or nonlinear envelope means Modulation of the sound with a distinct envelope behavior different than the monotonic decrease of the note. Can also include filter envelopes as well as dynamic envelopes.",
    "Percussive or sound with a percuss	means a loud non-harmonic sound at note onset, which is a fundamental part in string instrument, including acoustic guitar sound.",
    "Reverb	is a persistence of sound after it is produced.", # different from the dataset's discription
    "Tempo-synced measn rhythmic modulation of the sound to a fixed tempo."
]

def get_all_files(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            files_list.append(os.path.join(root, file))
    return files_list

def format_list(items):
    items = [s.replace("_", " ") for s in items]
    if len(items) == 0:
        return "with unknown sound quality."
    elif len(items) == 1:
        return f"with a {items[0]} sound quality."
    elif len(items) == 2:
        return f"with a {items[0]} and {items[1]} sound quality."
    else:
        return "with a "+", ".join(items[:-1]) + ", and " + items[-1] + " sound quality."

def raw_dataset_samples_single_list(files, f, t: float=None):
    annotations = json.load(f)
    for fpath in files:
        if not (fpath.endswith(".wav") or fpath.endswith(".mp3")):
            continue
        # data = resampler(lbs.to_mono(data).float().cuda())
        # data = lbs.resample(y=lbs.to_mono(data.T), orig_sr=sr, target_sr=22050, res_type='kaiser_fast')
        data, sr = sf.read(fpath)

        # 创建io.BytesIO对象
        buffer = io.BytesIO()
        # 将音频数据保存为二进制数据
        sf.write(buffer, data, sr, format='WAV', subtype='PCM_16')
        # 获取二进制数据
        binary_data = buffer.getvalue()

        key = os.path.splitext(fpath.split('/')[-1])[0]
        text = {"text": []}
        text['text'].append(f"a {annotations[key]["instrument_source_str"]} {annotations[key]["instrument_family_str"]} {format_list(annotations[key]["qualities_str"])}")
        text['text'].append(NOTE_QUA_LIST[annotations[key]["qualities"]==1])

        sample = {
            "__key__": key,
            "wav": binary_data,
            "json": json.dumps(text)
            }
        yield sample

def write_wds(root, save_root, data_list, info, start_shard: int = 0, t: float=None):
    #create .tar file
    assert(info in ["train","valid","test"]), "Wrong Training Label!"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(os.path.join(save_root, info)):
        os.makedirs(os.path.join(save_root, info))
    shard_num = 1+start_shard
    total_tmp1 = 0
    total_tmp2 = 1
    sizes_json = {}
    # if start_index == None:
    #     start_index=0
    #     max_num=start_index+max_num
    pid = str(os.getpid())
    while os.path.exists(os.path.join(save_root, info, pid+'-0.tar')):
        pid += 1
    # samples=raw_dataset_samples_single_list(data_list,aug,audio_dscrp,t)
    annotations = open(os.path.join(root, info, "examples.json"), "r")
    with wds.ShardWriter(pattern=os.path.join(save_root, info, pid+"-%d.tar"), maxcount=512, maxsize = 1e9, start_shard = start_shard) as sink:
        for sample in islice(raw_dataset_samples_single_list(data_list,annotations,t), 0, None):
            if shard_num==sink.shard:
                sink.write(sample)
                total_tmp1 = sink.total
            else: #if maximum conditions reached
                sizes_json[pid+'-'+str(shard_num-1)+".tar"]=sink.total-total_tmp2
                total_tmp2 = sink.total
                shard_num+=1
        sizes_json[pid+'-'+str(shard_num-1)+".tar"]=total_tmp1-total_tmp2+1
        ss = sink.shard
    annotations.close()
    #create sizes.json file
    if os.path.exists(os.path.join(root, info, "sizes.json")):
        with open(os.path.join(root, info, "sizes.json"), "r") as f:
            data = json.load(f)
            sizes_json = {**data, **sizes_json}
    with open(os.path.join(root, info, "sizes.json"), "w") as f:
        json.dump(sizes_json, f, indent = 4)
    # for key in sizes_json.keys():
    #     shared_dict[key] = sizes_json[key]
    return ss


if __name__ == "__main__":
    '''
    Params:
        root (str):  path of nsynth dataset in oringinal format, should contain three subfolder: train, valid, test
        save_path (str): where you would like to save the data
    '''
    root = '/datasets/nsynth'
    save_path = "/home/isaac/datasets/nsynth_web"
    cpu_num = 100
    
    process_args=[]
    train_list = get_all_files('/datasets/nsynth/train/audio')
    valid_list = get_all_files('/datasets/nsynth/valid/audio')
    test_list = get_all_files('/datasets/nsynth/test/audio')
    train_cpu = int(len(train_list)/(len(train_list)+len(valid_list)+len(test_list))*cpu_num)
    valid_cpu = int(len(valid_list)/(len(train_list)+len(valid_list)+len(test_list))*cpu_num)
    test_cpu = cpu_num-train_cpu-valid_cpu

    cpu_num = train_cpu + valid_cpu + test_cpu
    pool = multiprocessing.Pool(processes=cpu_num+3)
    # manager = multiprocessing.Manager()
    # shared_dict = {"train": manager.dict(), "valid": manager.dict(), "test": manager.dict()}


    r = list(range(0,len(train_list)-len(train_list)%train_cpu,len(train_list)//train_cpu))+[len(train_list)]
    r1 = copy.deepcopy(r);r1.pop()
    r2 = list(r[1:])
    for start, end in zip(r1, r2):
        process_args.append(
            {
                'root':root,
                'save_root':save_path, 
                'data_list':train_list[start:end], 
                'info':'train', 
                # 'shared_dict': shared_dict["train"],
                'start_shard': 0,
                # 'start_index' : start,
                # 'end_index': end,  
                't':None
            }
        )

    r = list(range(0,len(valid_list)-len(valid_list)%valid_cpu,len(valid_list)//valid_cpu))+[len(valid_list)]
    r1 = list(r);r1.pop()
    r2 = list(r[1:])
    for start, end in zip(r1, r2):
        process_args.append(
            {
                'root':root,
                'save_root':save_path, 
                'data_list':valid_list[start:end], 
                'info':'valid',
                # 'shared_dict': shared_dict["valid"],
                'start_shard' : 0,
                # 'start_index' : start,
                # 'end_index': end,  
                't':None
            }
        )

    r = list(range(0,len(test_list)-len(test_list)%test_cpu,len(test_list)//test_cpu))+[len(test_list)]
    r1 = list(r);r1.pop()
    r2 = list(r[1:])
    for start, end in zip(r1, r2):
        process_args.append(
            {
                'root':root,
                'save_root':save_path, 
                'data_list':test_list[start:end], 
                'info':'test',
                # 'shared_dict': shared_dict["test"],
                'start_shard': 0,
                # 'start_index' : start,
                # 'end_index': end,  
                't':None
            }
        )
        
    #test
    # write_wds(**process_args[0])

    for i, args in enumerate(process_args):
        pool.apply_async(write_wds, tuple(args.values()))
    pool.close()
    pool.join()

    #create sizes.json file
    # if os.path.exists(os.path.join(root, info, "sizes.json")):
    # for info in ["train","valid","test"]:
    #     with open(os.path.join(save_path, info, "sizes.json"), "w") as f:
    #         json.dump(dict(shared_dict[info]), f, indent = 4)

# TODO: 设置一个共享的全局变量保证不会同时重复写入
# root = "/home/isaac/datasets/nsynth"
# info = "train"
# with open(os.path.join(root, "nsynth-"+info, "examples.json"), "r") as f:
#     data = json.load(f)
#     text = {"text": []}
#     for key in data.keys():
#         text['text'].append(f"a {data[key]["instrument_source_str"]} {data[key]["instrument_source_str"]}"\
#             + ("with a "+data[key]["qualities_str"] + " sound· quality.") if data[key]["qualities_str"]\
#                 else "with unknown sound quality.")
#         text['text'].append(NOTE_QUA_LIST[data[key]["qualities"]==1])