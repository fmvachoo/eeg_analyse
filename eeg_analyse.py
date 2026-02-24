import pandas as pd
import numpy as np
import os
import glob

DATA_FOLDER = "D:\WORK\egg_analyse\EEG RAW DATA (INFOPOVODI)\EEG RAW DATA (INFOPOVODI)"

STIMULI = [
    'VK_JAPAN_INFO',   'VK_JAPAN_COM',   'VK_JAPAN_THR',
    'TG_JAPAN_INFO',   'TG_JAPAN_COM',   'TG_JAPAN_THR',
    'TG_MUSK_INFO',    'TG_MUSK_COM',    'TG_MUSK_THR',
    'VK_MUSK_INFO',    'VK_MUSK_COM',    'VK_MUSK_THR',
    'VK_BORISOV_INFO', 'VK_BORISOV_COM', 'VK_BORISOV_THR',
    'TG_BORISOV_INFO', 'TG_BORISOV_COM', 'TG_BORISOV_THR',
    'TG_EGE_INFO',     'TG_EGE_COM',     'TG_EGE_THR_1', 'TG_EGE_THR_2',
    'VK_EGE_INFO',     'VK_EGE_COM'
]

# в файлах написано MASK вместо MUSK, исправил
TYPO_FIX = {
    'TG_MASK_INFO': 'TG_MUSK_INFO',
    'VK_MASK_INFO': 'VK_MUSK_INFO',
}

# 14 ЭЭГ-каналов гарнитуры EPOC X
EEG_CHANNELS = [
    'EEG.AF3', 'EEG.F7',  'EEG.F3',  'EEG.FC5',
    'EEG.T7',  'EEG.P7',  'EEG.O1',  'EEG.O2',
    'EEG.P8',  'EEG.T8',  'EEG.FC6', 'EEG.F4',
    'EEG.F8',  'EEG.AF4'
]

# Имена каналов без префикса EEG
CH_NAMES = [c.replace('EEG.', '') for c in EEG_CHANNELS]

# Эмоциональные метрики
EMOTION_COLS = [
    'PM.Attention.Scaled', 'PM.Engagement.Scaled',
    'PM.Excitement.Scaled', 'PM.Stress.Scaled',
    'PM.Relaxation.Scaled', 'PM.Interest.Scaled',
    'PM.Focus.Scaled'
]

SFREQ = 128     # частота дискретизации
PRE_MS = 200   # мс до стимула
POST_MS = 1000  # мс после стимула

# Номера всех респондентов: 'pilot' = 0, остальные 1-16
RESPONDENTS = ['0', '1', '2', '3', '4', '5', '6', '7',
               '8', '9', '10', '11', '12', '13', '14', '15', '16']

# Загрузка файлов
def load_raw(filepath):
    """
    Загрузка файла N_RAW_DATA
    """
    df = pd.read_csv(filepath, sep=';', skiprows=1, header=0)
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).reset_index(drop=True)
    return df

def load_markers(filepath):
    """
    Загрузка файлов маркеров
    """
    return pd.read_csv(filepath)

def find_files(folder, respondent_id):
    """
    Ищет файлы для респондента. 
    Вернёт (raw_path, marker_path) или (None, None) если не нашли.
    """
    r = str(respondent_id)

    raw_matches = glob.glob(os.path.join(folder, f"{r}_RAW DATA.csv"))

    marker_matches = glob.glob(os.path.join(folder, f"{r}_*intervalMarker.csv"))

    raw    = raw_matches[0]    if raw_matches    else None
    marker = marker_matches[0] if marker_matches else None

    return raw, marker

def get_stimulus_events(markers_df):
    """
    Извлекает строки, исправляет опечатки.
    Возвращает DataFrame: stimulus_name | latency
    """
    all_know_names = STIMULI + list(TYPO_FIX.keys())
    
    stim = markers_df[markers_df['type'].isin(all_know_names)].copy()
    
    stim['stimulus_name'] = stim['type'].replace(TYPO_FIX)
    
    return stim [['stimulus_name', 'latency']].reset_index(drop=True)

def latency_to_sample_index(raw_df, latency_sec):
    """
    Конвертирует latency (секунды от начала записи)
    в индекс строки в raw_df
    """
    start_time = raw_df['Timestamp'].iloc[0]
    stim_abs_time = start_time + latency_sec

    timestamps = raw_df['Timestamp'].values

    idx = np.searchsorted(timestamps, stim_abs_time)
    idx = min(idx, len(timestamps) - 1)
    return idx

def extract_epoch_eeg(raw_df, sample_idx):
    """
    Вырезает часть ЭЭГ вокруг стимула
    """

    pre_samples = int(PRE_MS * SFREQ / 1000)
    post_samples = int(POST_MS * SFREQ / 1000)

    start   = sample_idx - pre_samples
    end     = sample_idx + post_samples

    # если стимул слишком близко к краю файла
    if start < 0 or end > len(raw_df):
        return None

    # .values: pandas DataFrame -> numpy array (быстрее для вычислений)
    return raw_df.iloc[start:end][EEG_CHANNELS].values  # shape: (153, 14)

def baseline_correct(epoch):
    """
    Коррекция вычитанием среднего
    по предстимульному интервалу
    """

    pre_samples = int(PRE_MS * SFREQ / 1000)
    
    baseline_mean = epoch[:pre_samples].mean(axis=0)

    return epoch - baseline_mean


# ТЕСТ НА ОДНОМ РЕСПОНДЕНТЕ

raw_path, marker_path = find_files(DATA_FOLDER, '1')
raw = load_raw(raw_path)
markers = load_markers(marker_path)
events = get_stimulus_events(markers)

print("Найденные стимулы для респондента 1:")
print(events)
print(f"\nВсего стимулов: {len(events)}")
print(f"Нашли все 24? {set(events['stimulus_name']) == set(STIMULI)}")

# Тест на одном стимуле
first_stim = events.iloc[0]
idx = latency_to_sample_index(raw, first_stim['latency'])
epoch = extract_epoch_eeg(raw, idx)
epoch_bc = baseline_correct(epoch)

print(f"\nСтимул: {first_stim['stimulus_name']}")
print(f"Индекс в raw_df: {idx}")
print(f"Форма эпохи: {epoch.shape}")  # должно быть (153, 14)
print(f"Форма после baseline: {epoch_bc.shape}")
print(f"Среднее baseline до коррекции: {epoch[:25].mean():.4f} мкВ")
print(f"Среднее baseline после коррекции: {epoch_bc[:25].mean():.6f} мкВ")  # должно быть ~0
