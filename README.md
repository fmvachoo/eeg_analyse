# 🧠 EEG ERP Analysis — Восприятие инфоповодов

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange?logo=numpy)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-red)](https://matplotlib.org)

Анализ ЭЭГ-данных методом усреднённых вызванных потенциалов (ERP) для оценки
когнитивного восприятия новостных сообщений из социальных сетей ВКонтакте и Telegram.

---

## 📋 Описание исследования

Участникам предъявлялись новостные сообщения из двух платформ (ВКонтакте и Telegram)
по четырём темам. Каждое сообщение представлено в трёх форматах подачи.
ЭЭГ записывалась с помощью гарнитуры **Emotiv EPOC X** (14 каналов, 128 Гц).

**Итого:** 24 стимула × 16 респондентов

---

## 🗂️ Структура репозитория
├── results_ERP/               # ERP-графики для всех 24 стимулов

│   ├── ERP_VK_JAPAN_INFO.png

│   ├── ERP_TG_JAPAN_INFO.png

│   └── ...

├── eeg_analyse.py             # Основной скрипт анализа

└── README.md

---

## ⚙️ Методы

### Запись ЭЭГ

- Устройство: **Emotiv EPOC X**
- Каналы: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
- Частота дискретизации: **128 Гц**

### Обработка сигнала


Сырые данные
↓
Поиск меток стимулов (intervalMarker)
↓
Нарезка эпох: −200 мс ... +1000 мс относительно стимула
↓
Baseline-коррекция (вычитание среднего за −200...0 мс)
↓
Усреднение по респондентам → ERP

---

## 📊 Результаты

### VK — Японский инфоповод

| INFO | COM | THR |
|------|-----|-----|
| ![](results_ERP/ERP_VK_JAPAN_INFO.png) | ![](results_ERP/ERP_VK_JAPAN_COM.png) | ![](results_ERP/ERP_VK_JAPAN_THR.png) |

### TG — Японский инфоповод

| INFO | COM | THR |
|------|-----|-----|
| ![](results_ERP/ERP_TG_JAPAN_INFO.png) | ![](results_ERP/ERP_TG_JAPAN_COM.png) | ![](results_ERP/ERP_TG_JAPAN_THR.png) |

### VK — Маск

| INFO | COM | THR |
|------|-----|-----|
| ![](results_ERP/ERP_VK_MUSK_INFO.png) | ![](results_ERP/ERP_VK_MUSK_COM.png) | ![](results_ERP/ERP_VK_MUSK_THR.png) |

### TG — Маск

| INFO | COM | THR |
|------|-----|-----|
| ![](results_ERP/ERP_TG_MUSK_INFO.png) | ![](results_ERP/ERP_TG_MUSK_COM.png) | ![](results_ERP/ERP_TG_MUSK_THR.png) |

### VK — Борисов

| INFO | COM | THR |
|------|-----|-----|
| ![](results_ERP/ERP_VK_BORISOV_INFO.png) | ![](results_ERP/ERP_VK_BORISOV_COM.png) | ![](results_ERP/ERP_VK_BORISOV_THR.png) |

### TG — Борисов

| INFO | COM | THR |
|------|-----|-----|
| ![](results_ERP/ERP_TG_BORISOV_INFO.png) | ![](results_ERP/ERP_TG_BORISOV_COM.png) | ![](results_ERP/ERP_TG_BORISOV_THR.png) |

### VK — ЕГЭ

| INFO | COM |
|------|-----|
| ![](results_ERP/ERP_VK_EGE_INFO.png) | ![](results_ERP/ERP_VK_EGE_COM.png) |

### TG — ЕГЭ

| INFO | COM | THR 1 | THR 2 |
|------|-----|-------|-------|
| ![](results_ERP/ERP_TG_EGE_INFO.png) | ![](results_ERP/ERP_TG_EGE_COM.png) | ![](results_ERP/ERP_TG_EGE_THR_1.png) | ![](results_ERP/ERP_TG_EGE_THR_2.png) |

---

### Установка зависимостей

````bash
pip install pandas numpy matplotlib

