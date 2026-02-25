import pandas as pd
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

DATA_FOLDER = "D:\WORK\egg_analyse\EEG RAW DATA (INFOPOVODI)\EEG RAW DATA (INFOPOVODI)"

# ДИАГНОСТИКА — запусти отдельно, чтобы понять в чём проблема у респондента 2
raw_path_2, _ = find_files(DATA_FOLDER, '2')
print(f"Файл: {raw_path_2}")

# Читаем как есть, без исправлений
df_test = pd.read_csv(raw_path_2, sep=';', skiprows=1, header=0)

print(f"\nПервые 5 названий столбцов:")
for col in list(df_test.columns[:5]):
    # repr() показывает скрытые символы — увидим \ufeff если он есть
    print(f"  {repr(col)}")