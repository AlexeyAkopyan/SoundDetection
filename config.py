import math


data_path = "UrbanSound8K"
save_path = '.'
sr = 44100
model_path = 'model.json'

n_window = 2048
hop_length = 511
n_mels = 64
max_len_seconds = 10.
max_frames = math.ceil(max_len_seconds * sr / hop_length)
pooling_time_ratio = 8

f_min = 0.
f_max = 22050.
save_log_feature = False

classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']