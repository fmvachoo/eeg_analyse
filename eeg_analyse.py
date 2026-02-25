from pickletools import stackslice
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import os
import glob
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

DATA_FOLDER = "D:\WORK\egg_analyse\EEG RAW DATA (INFOPOVODI)\EEG RAW DATA (INFOPOVODI)"
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "results_ERP")

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

SFREQ = 128                                     # частота дискретизации
PRE_MS = 200                                    # мс до стимула
POST_MS = 1000                                  # мс после стимула
time_ms = np.arange(-25, 128) / SFREQ * 1000    # ось времени в мс
COLORS = plt.cm.tab20.colors[:14]               # цвета для каналов - 14 цветов

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

def collect_all_epochs(folder, respondents):
    """
    Проходим по всем ремпондентам и собираем эпохи
    """
    all_epochs = {stim: [] for stim in STIMULI}

    for resp_id in respondents:
        raw_path, marker_path = find_files(folder, resp_id)

        if raw_path is None or marker_path is None:
            print(f"  [!] Респондент {resp_id}: файлы не найдены")
            continue

        print(f"Загрузка респондента {resp_id}.. ", end = ' ')

        try:
            raw     = load_raw(raw_path)
            markers = load_markers(marker_path)
            events  = get_stimulus_events(markers)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        ok_count = 0
        for _, row in events.iterrows():
            stim_name = row['stimulus_name']
            latency = row['latency']

            idx = latency_to_sample_index(raw, latency)
            epoch = extract_epoch_eeg(raw, idx)

            if epoch is None:
                continue

            epoch_bc = baseline_correct(epoch)
            all_epochs[stim_name].append(epoch_bc)
            ok_count += 1

        print(f"OK ({ok_count}/24 stimulus)")
    return all_epochs

def compute_erp(epochs_list):
    """
    Усредняет эпохи по всем респондентам
    """
    if len(epochs_list) == 0:
        return None

    stacked = np.stack(epochs_list, axis=0)

    return stacked.mean(axis=0)
    
def plot_erp(stim_name, erp, n_respondents, output_folder):
    """
    Строит ERP-график для одного стимула
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    for ch_idx, ch_name in enumerate(CH_NAMES):
        ax.plot(
            time_ms,
            erp[:, ch_idx],
            color=COLORS[ch_idx],
            linewidth=1.2,
            label=ch_name
        )

    ax.axvline(x=0, color='red', linewidth=1.5, linestyle='--', label='Stimul')
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)
    ax.axvspan(time_ms[0], 0, alpha=0.08, color='gray', label='Baseline')

    ax.set_title(f'ERP - {stim_name} (n={n_respondents})', fontsize=14)
    ax.set_xlabel('Время (мс)', fontsize=12)
    ax.set_ylabel('Амплитуда (мкВ)', fontsize=12)
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Подписи делений оси X каждые 100мс
    ax.set_xticks(range(-200, 1001, 100))

    plt.tight_layout()

    # Сохраняем в файл
    save_path = os.path.join(output_folder, f"ERP_{stim_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)   # закрываем, чтобы не накапливать в памяти
    return save_path

print("Собираем эпохи по всем респондентам...")
all_epochs = collect_all_epochs(DATA_FOLDER, RESPONDENTS)

print("\nКоличество эпох на стимул:")
for stim, epochs in all_epochs.items():
    print(f"  {stim}: {len(epochs)} респондентов")

erp_dict = {}
for stim, epochs in all_epochs.items():
    erp_dict[stim] = compute_erp(epochs)

print("ERP посчитаны для всех стимулов")

# Строим графики
print("\nСтроим ERP-графики...")
for stim_name, erp in erp_dict.items():
    if erp is None:
        print(f"  [!] {stim_name}: нет данных, пропускаем")
        continue
    n = len(all_epochs[stim_name])
    path = plot_erp(stim_name, erp, n, OUTPUT_FOLDER)
    print(f"  ✓ {stim_name} → {os.path.basename(path)}")

print(f"\nГрафики сохранены в: {OUTPUT_FOLDER}")

