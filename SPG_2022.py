# -*- coding: utf-8 -*-
"""
Обработка минутных геомагнитных данных обсерватории SPG (Санкт-Петербург).
Формат: IAGA-2002.
Все имена файлов и директорий генерируются на основе префикса dataset_prefix.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------------------------
# 🔑 ЦЕНТРАЛЬНАЯ НАСТРОЙКА: задайте префикс один раз
# -------------------------------------------------
dataset_prefix = "spg2022"          # ←←← МЕНЯТЬ ТОЛЬКО ЭТУ СТРОКУ
station_code = "SPG"                # Код обсерватории (для структуры пути)
year = "2022"                       # Год (для структуры пути)

# Автоматическая генерация всех зависимых имён
file_pattern = f"{station_code}/{year}/{dataset_prefix}*.min"
output_plot_dir = f"plots_{dataset_prefix}"
output_csv_file = f"{dataset_prefix}_processed_full.csv"

# Настройка графиков
plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})

# ----------------------------
# 1. СОЗДАНИЕ ДИРЕКТОРИИ ДЛЯ ГРАФИКОВ
# ----------------------------
os.makedirs(output_plot_dir, exist_ok=True)
print(f"📁 Директория для графиков: {os.path.abspath(output_plot_dir)}")

# ----------------------------
# 2. ПОИСК ФАЙЛОВ
# ----------------------------
all_files = sorted(glob.glob(file_pattern))

if not all_files:
    raise FileNotFoundError(
        f"Файлы не найдены по шаблону: '{file_pattern}'.\n"
        f"Убедитесь, что папка {station_code}/{year}/ существует и содержит файлы {dataset_prefix}*.min."
    )

print(f"📂 Найдено файлов: {len(all_files)}")
for f in all_files[:3]:
    print(f"  → {f}")

# ----------------------------
# 3. ПАРСИНГ ФАЙЛОВ IAGA-2002
# ----------------------------

def parse_iaga2002_min_file(filepath):
    """Парсит IAGA-2002 файл с минутными вариационными данными."""
    metadata = {}
    data_started = False
    records = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # --- Сбор метаданных ---
            if stripped.startswith('Format'):
                metadata['Format'] = stripped.split('Format', 1)[1].strip()
            elif stripped.startswith('Station Name'):
                metadata['Station_Name'] = stripped.split('Station Name', 1)[1].strip()
            elif stripped.startswith('IAGA Code'):
                metadata['IAGA_Code'] = stripped.split('IAGA Code', 1)[1].strip()
            elif stripped.startswith('Geodetic Latitude'):
                try:
                    metadata['Latitude'] = float(stripped.split('Geodetic Latitude', 1)[1].strip())
                except:
                    pass
            elif stripped.startswith('Geodetic Longitude'):
                try:
                    metadata['Longitude'] = float(stripped.split('Geodetic Longitude', 1)[1].strip())
                except:
                    pass
            elif stripped.startswith('Reported'):
                metadata['Components'] = stripped.split('Reported', 1)[1].strip()
            elif stripped.startswith('Data Type'):
                metadata['Data_Type'] = stripped.split('Data Type', 1)[1].strip()

            # --- Начало данных ---
            if stripped.startswith('DATE') and 'TIME' in stripped:
                data_started = True
                continue

            # --- Чтение данных ---
            if data_started and stripped[0].isdigit():
                try:
                    parts = stripped.split()
                    if len(parts) < 7:
                        continue
                    dt = datetime.strptime(f"{parts[0]} {parts[1]}", '%Y-%m-%d %H:%M:%S.%f')
                    x = np.nan if parts[3] == '99999.00' else float(parts[3])
                    y = np.nan if parts[4] == '99999.00' else float(parts[4])
                    z = np.nan if parts[5] == '99999.00' else float(parts[5])
                    f_val = np.nan if parts[6] == '99999.00' else float(parts[6])
                    records.append([dt, x, y, z, f_val])
                except (ValueError, IndexError):
                    continue

    df = pd.DataFrame(records, columns=['DateTime', 'X', 'Y', 'Z', 'F'])
    df.set_index('DateTime', inplace=True)
    return df, metadata

# ----------------------------
# 4. ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ДАННЫХ
# ----------------------------

print("\n🔍 Чтение и парсинг файлов...")
all_dfs = []
global_meta = {}

for i, fp in enumerate(all_files):
    df_part, meta = parse_iaga2002_min_file(fp)
    all_dfs.append(df_part)
    if i == 0:
        global_meta = meta

df_raw = pd.concat(all_dfs).sort_index()
df_raw = df_raw[~df_raw.index.duplicated(keep='first')]

if len(df_raw) == 0:
    raise ValueError("Не загружено ни одной записи. Проверьте формат файлов.")

print(f"✅ Загружено записей: {len(df_raw)}")
print(f"📅 Период: с {df_raw.index.min()} по {df_raw.index.max()}")

print("\n📄 Метаданные (из первого файла):")
for k, v in global_meta.items():
    print(f"  {k}: {v}")

# ----------------------------
# 5. РАЗВЕДОЧНЫЙ АНАЛИЗ (EDA)
# ----------------------------

print("\n📊 Разведочный анализ:")
print(df_raw.describe())
print("\nПропуски (%):")
print((df_raw.isnull().mean() * 100).round(3))

# График 1: Исходные данные
plt.figure(figsize=(16, 10))
for i, comp in enumerate(['X', 'Y', 'Z', 'F'], 1):
    plt.subplot(2, 2, i)
    plt.plot(df_raw.index, df_raw[comp], 'steelblue', linewidth=0.6, alpha=0.8)
    plt.title(f'Исходные: {comp}')
    plt.ylabel('нТл')
    plt.grid(True, linestyle='--', alpha=0.5)
plt.suptitle(f'Геомагнитные данные {station_code} — ИСХОДНЫЕ ({dataset_prefix})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir, f'{dataset_prefix}_raw.png'), dpi=150, bbox_inches='tight')
plt.show()

# ----------------------------
# 6. ПРЕДОБРАБОТКА
# ----------------------------

df_proc = df_raw.copy()
df_proc = df_proc.interpolate(method='linear')
df_proc = df_proc.bfill().ffill()

print("\n🧹 Предобработка завершена. Пропуски после обработки (%):")
print((df_proc.isnull().mean() * 100).round(6))

# График 2: Обработанные данные
plt.figure(figsize=(16, 10))
for i, comp in enumerate(['X', 'Y', 'Z', 'F'], 1):
    plt.subplot(2, 2, i)
    plt.plot(df_proc.index, df_proc[comp], 'forestgreen', linewidth=0.7)
    plt.title(f'Обработанные: {comp}')
    plt.ylabel('нТл')
    plt.grid(True, linestyle='--', alpha=0.5)
plt.suptitle(f'Геомагнитные данные {station_code} — ПОСЛЕ ПРЕДОБРАБОТКИ ({dataset_prefix})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir, f'{dataset_prefix}_processed.png'), dpi=150, bbox_inches='tight')
plt.show()

# График 3: Сравнение
gap_mask = df_raw.isnull().any(axis=1)
if gap_mask.any():
    start_gap = df_raw.index[gap_mask].min()
    plot_start = start_gap - pd.Timedelta(hours=2)
    plot_end = start_gap + pd.Timedelta(hours=12)
else:
    plot_start = df_raw.index[0]
    plot_end = plot_start + pd.Timedelta(days=7)

plt.figure(figsize=(16, 10))
for i, comp in enumerate(['X', 'Y', 'Z', 'F'], 1):
    plt.subplot(2, 2, i)
    plt.plot(df_raw.loc[plot_start:plot_end].index, df_raw.loc[plot_start:plot_end, comp],
             'ro', markersize=2, label='Исходные', alpha=0.7)
    plt.plot(df_proc.loc[plot_start:plot_end].index, df_proc.loc[plot_start:plot_end, comp],
             'forestgreen', linewidth=1.5, label='После обработки')
    plt.title(f'Сравнение: {comp}')
    plt.ylabel('нТл')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
plt.suptitle(f'Сравнение данных ({dataset_prefix})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir, f'{dataset_prefix}_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

# ----------------------------
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ----------------------------

df_proc.to_csv(output_csv_file)
print(f"\n💾 Сохранено: {output_csv_file}")
print(f"🖼️  Графики в: {output_plot_dir}")
print("\n✅ Анализ завершён.")