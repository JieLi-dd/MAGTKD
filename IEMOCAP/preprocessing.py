import csv

def split(session):
    final_data = []
    split_session = []
    for line in session:
        split_session.append(line)
        final_data.append(split_session[:])
    return final_data

def preprocessing(data_path, split_type):
    session_dataset = []
    session = []
    speaker_set = []

    with open(data_path, 'r') as f:
        rdr = csv.reader(f)

        header = next(rdr)  # 读取表头
        utt_idx = header.index('Utterance')
        speaker_idx = header.index('Speaker')
        emo_idx = header.index('Emotion')
        sess_idx = header.index('Dialogue_ID')
        uttid_idx = header.index('Utterance_ID')
        wav_idx = header.index('Wav_Path')
        video_idx = header.index('Video_Path')
        start_idx = header.index('Start_Time')
        end_idx = header.index('End_Time')

        pre_sess = 'start'
        for line in rdr:
            utt = line[utt_idx]
            speaker = line[speaker_idx][line[speaker_idx].rfind('_')+1]
            emotion = line[emo_idx]
            sess = line[sess_idx]
            uttid = line[uttid_idx]
            video_path = line[video_idx]
            wav_path = line[wav_idx]
            start_time = line[start_idx]
            end_time = line[end_idx]

            if speaker not in speaker_set:
                speaker_set.append(speaker)
            uniq_speaker = speaker_set.index(speaker)

            if pre_sess == 'start' or sess == pre_sess:
                session.append([uniq_speaker, utt, wav_path, video_path, start_time, end_time, emotion, split_type, sess, uttid])
            else:
                session_dataset.extend(split(session))
                session = [[uniq_speaker, utt, wav_path, video_path, start_time, end_time, emotion, split_type, sess, uttid]]
                speaker_set = []

            pre_sess = sess

    # 处理最后一个 session
    session_dataset.extend(split(session))

    """
        return struction：
        session_dataset[
            0000
            0001[
                0[uniq_speaker, utt, wav_path, video_path, start_time, end_time, emotion, split_type, sess, uttid]
                1
            ]
            0002
            ....
        ]
    """

    return session_dataset