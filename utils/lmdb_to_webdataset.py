# import lmdb
# import webdataset as wds
# import os
# import io
# import librosa
# import soundfile as sf
# from audio_example import AudioExample
# import json
# # 设置路径
# lmdb_path = "/datasets/urmp_lmdb_FullTrack"
# output_dir = "/datasets/urmp_webdataset"
# output_pattern = os.path.join(output_dir, "data-%06d.tar")

# # 创建 WebDataset 的 ShardWriter
# with wds.ShardWriter(output_pattern, maxcount=1000) as sink:
#     # 打开 LMDB 环境
#     env = lmdb.open(
#         lmdb_path,
#         lock=False,
#         readonly=True,
#         readahead=False,
#         map_async=False,
#     )
#     with env.begin() as txn:
#         keys = list(txn.cursor().iternext(values=False))

#     # with env.begin() as txn:
#         for key in keys:
#             ae = AudioExample(txn.get(key))
#             waveform = librosa.resample(ae.get('waveform'), orig_sr=44100, target_sr=48000)
#             buffer = io.BytesIO()
#             # 将音频数据保存为二进制数据
#             sf.write(buffer, waveform, 48000, format='WAV', subtype='PCM_16')
#             # 获取二进制数据
#             binary_data = buffer.getvalue()

#             texts = ae.get_metadata()['texts']
#             text = {}
#             text["text"] = texts
#                 # 创建样本字典
#             sample = {
#                 "__key__": key.decode("utf-8"),
#                 "wav": binary_data,
#                 "data": json.dump(text)  # 假设 value 是二进制数据
#                 # 如果有其他数据，如图像或标签，可以继续添加
#             }
#             # 写入 WebDataset
#             sink.write(sample)

import lmdb
import webdataset as wds
import os
import io
import librosa
import soundfile as sf
from audio_example import AudioExample
import json
import concurrent.futures

# 设置路径
lmdb_path = "/datasets/urmp_lmdb_FullTrack"
output_dir = "/datasets/urmp_webdataset"
output_pattern = os.path.join(output_dir, "data-%06d.tar")

def process_key(key, txn):
    ae = AudioExample(txn.get(key))
    waveform = librosa.resample(ae.get('waveform'), orig_sr=44100, target_sr=48000)
    
    buffer = io.BytesIO()
    sf.write(buffer, waveform, 48000, format='WAV', subtype='PCM_16')
    binary_data = buffer.getvalue()

    texts = ae.get_metadata()['texts']
    text = {"text": texts}

    sample = {
        "__key__": key.decode("utf-8"),
        "wav": binary_data,
        "data": json.dumps(text)  # 使用 json.dumps 进行正确的序列化
    }
    
    return sample

# 创建 WebDataset 的 ShardWriter
with wds.ShardWriter(output_pattern, maxcount=1000) as sink:
    # 打开 LMDB 环境
    env = lmdb.open(
        lmdb_path,
        lock=False,
        readonly=True,
        readahead=False,
        map_async=False,
    )
    
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))

    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        futures = {executor.submit(process_key, key, txn): key for key in keys}

        for future in concurrent.futures.as_completed(futures):
            sample = future.result()
            # 写入 WebDataset
            sink.write(sample)