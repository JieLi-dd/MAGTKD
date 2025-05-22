import os
import csv
import pickle
import numpy as np
from tqdm import tqdm
import cv2
from transformers import AutoImageProcessor

# 加载预训练的视频特征提取模型
video_processor = AutoImageProcessor.from_pretrained("../pretrained_model/videomae-base")

# 提取视频特征的函数
def get_video(feature_extractor, path, start_time, end_time):
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    time_start = max(float(start_time) - 1, 0)
    time_end = float(end_time) + 1
    length = fps * (time_end - time_start)
    frames = []
    step = length // 8
    count = 0

    try:
        while video.isOpened():
            ret, image = video.read()
            if not ret:
                break
            frame_pos = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            if time_start * fps <= frame_pos <= time_end * fps:
                count += 1
                if count % step == 0:
                    frames.append(image)
        video.release()

        if len(frames) >= 8:
            inputs = feature_extractor(frames[:8], return_tensors="pt")
            return inputs["pixel_values"][0].numpy()
        else:
            lack = 8 - len(frames)
            extend_frames = [frames[-1].copy() for _ in range(lack)]
            frames.extend(extend_frames)
            inputs = feature_extractor(frames[:8], return_tensors="pt")
            return inputs["pixel_values"][0].numpy()

    except Exception as e:
        print(f"Error processing video: {path}, start_time: {start_time}, end_time: {end_time}, error: {e}")
        return None

if __name__ == '__main__':
    data_path = "../datasets/IEMOCAP"

    train_path = os.path.join(data_path, "IEMOCAP_train.csv")
    dev_path = os.path.join(data_path, "IEMOCAP_dev.csv")
    test_path = os.path.join(data_path, "IEMOCAP_test.csv")

    output_dir = "./feature/video"
    os.makedirs(output_dir, exist_ok=True)

    # 按数据集分类存储
    for path, dataset in zip([train_path, dev_path, test_path], ['train', 'dev', 'test']):
        dataset_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        with open(path, 'r') as f:
            rdr = csv.reader(f)
            header = next(rdr)  # 读取表头
            utt_idx = header.index('Utterance')
            speaker_idx = header.index('Speaker')
            emo_idx = header.index('Emotion')
            sess_idx = header.index('Dialogue_ID')
            uttid_idx = header.index('Utterance_ID')
            video_idx = header.index('Video_Path')
            start_idx = header.index('Start_Time')
            end_idx = header.index('End_Time')

            for line in tqdm(rdr, desc=f"Processing {dataset} set"):
                uttid = line[uttid_idx]
                sess = line[sess_idx]
                video_path = line[video_idx]
                start_time = line[start_idx]
                end_time = line[end_idx]

                # 提取视频特征
                video_feature = get_video(video_processor, video_path, start_time, end_time)
                if video_feature is None:
                    continue

                # 按 (sess, uttid) 命名文件
                file_name = f"{sess}_{uttid}.npy"
                file_path = os.path.join(dataset_dir, file_name)

                # 存储为 npy 格式
                np.save(file_path, video_feature)

    print("Video feature extraction and saving completed.")
