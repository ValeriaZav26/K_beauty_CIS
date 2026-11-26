"""
utils.py — общие функции проекта K-beauty в СНГ (владелец: роль B).
Используют все три ноутбука: 01_market (A), 02_demand (B), 03_trends (C).

    from utils import load_expanded, zero_share, seasonal_index, time_split, \
        load_calendar, COUNTRY_MAP, PALETTE, raw_path, clean_path

Договорённости из ТЗ v4 (раздел 16):
- месяц всегда строкой 'YYYY-MM';
- страны — коды iso2 (RU, KZ, …);
- в файлах Trends *_expanded одна строка = одна единица индекса;
  месяц с 1 строкой = индекс 0 (16.7);
- везде конец периода — 2026-10 (неполный 2026-11 отрезаем);
- random_state = 42.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
LAST_MONTH = '2026-10'
SEARCH_START = '2020-01'

# ---------------------------------------------------------------- пути
ROOT = Path(__file__).resolve().parent


def raw_path(name):
    """Путь к сырому файлу: data/raw/<name>, а пока структуры нет — корень репозитория."""
    p = ROOT / 'data' / 'raw' / name
    return p if p.exists() else ROOT / name


def clean_path(name):
    """Путь к чистому файлу в data/clean/ (папка создаётся при необходимости)."""
    d = ROOT / 'data' / 'clean'
    d.mkdir(parents=True, exist_ok=True)
    return d / name


# ---------------------------------------------------------------- страны и цвета (ТЗ 16.3)
COUNTRY_MAP = pd.DataFrame([
    # iso2, iso3, название в таможне, русское название, группа, цвет
    ('RU', 'RUS', '러시아 연방', 'Россия', 'RU', '#D55E00'),
    ('KZ', 'KAZ', '카자흐스탄', 'Казахстан', 'other_CIS', '#0072B2'),
    ('KG', 'KGZ', '키르기스스탄', 'Кыргызстан', 'other_CIS', '#009E73'),
    ('UZ', 'UZB', '우즈베키스탄', 'Узбекистан', 'other_CIS', '#E69F00'),
    ('AM', 'ARM', '아르메니아', 'Армения', 'other_CIS', '#999999'),
    ('AZ', 'AZE', '아제르바이잔', 'Азербайджан', 'other_CIS', '#999999'),
    ('GE', 'GEO', '조지아', 'Грузия', 'other_CIS', '#999999'),
    ('TJ', 'TJK', '타지크', 'Таджикистан', 'other_CIS', '#999999'),
    ('TM', 'TKM', '투르크멘', 'Туркменистан', 'other_CIS', '#999999'),
    ('MD', 'MDA', '몰도바', 'Молдова', 'other_CIS', '#999999'),
    ('BY', 'BLR', '벨라루스', 'Беларусь', 'BY', '#CC79A7'),
    ('MN', 'MNG', '몽골', 'Монголия', 'MN', '#56B4E9'),
    ('CN', 'CHN', '중국', 'Китай', 'CN', '#7F7F7F'),
], columns=['iso2', 'iso3', 'name_kr', 'country_ru', 'group', 'color'])
WORLD_EX = {'iso2': 'WX', 'country_ru': 'Мир без СНГ и Китая', 'group': 'world_ex', 'color': '#000000'}
NOT_COUNTRIES = {'기타국', '우크라이나'}   # «прочие страны» и Украина не входят в группы

KR_TO_ISO2 = dict(zip(COUNTRY_MAP.name_kr, COUNTRY_MAP.iso2))
ISO2_TO_RU = dict(zip(COUNTRY_MAP.iso2, COUNTRY_MAP.country_ru)) | {'WX': WORLD_EX['country_ru']}
PALETTE = dict(zip(COUNTRY_MAP.iso2, COUNTRY_MAP.color)) | {'WX': WORLD_EX['color']}


# ---------------------------------------------------------------- месяцы
def month_range(start=SEARCH_START, end=LAST_MONTH):
    """Список месяцев 'YYYY-MM' от start до end включительно."""
    return pd.period_range(start, end, freq='M').astype(str).tolist()


# ---------------------------------------------------------------- Google Trends *_expanded
def load_expanded(path, keys, start=SEARCH_START, end=LAST_MONTH):
    """
    Восстанавливает индекс Google Trends из файла *_expanded.
    keys — колонки, задающие ряд (например ['geo', 'ingredient']).
    Индекс = число строк в месяце; 1 строка = индекс 0 (ТЗ 16.7);
    месяцы без строк тоже 0. Возвращает keys + month + value + rows.
    """
    d = pd.read_csv(path, encoding='utf-8-sig')
    d = d[(d['month'] >= start) & (d['month'] <= end)]
    n = d.groupby(keys + ['month']).size().rename('rows').reset_index()
    series = n[keys].drop_duplicates()
    grid = series.merge(pd.DataFrame({'month': month_range(start, end)}), how='cross')
    out = grid.merge(n, on=keys + ['month'], how='left').fillna({'rows': 0})
    out['rows'] = out['rows'].astype(int)
    out['value'] = np.where(out['rows'] <= 1, 0, out['rows'])
    return out.sort_values(keys + ['month']).reset_index(drop=True)


def zero_share(s):
    """Доля месяцев с индексом ≤ 1 (для _expanded это «1 строка», для остальных — 0 или 1)."""
    s = pd.Series(s)
    return float((s <= 1).mean()) if len(s) else np.nan


# ---------------------------------------------------------------- сезонность
def seasonal_index(s, months=None):
    """
    Сезонный профиль ряда: среднее по календарному месяцу от (значение / среднее своего года) × 100.
    100 = обычный месяц, 120 = на 20% выше среднего года.
    s — Series с индексом 'YYYY-MM' (или передать months отдельно).
    Берутся только полные годы (12 месяцев), чтобы неполный год не искажал профиль.
    """
    s = pd.Series(np.asarray(s, dtype=float), index=months if months is not None else s.index)
    df = pd.DataFrame({'v': s.values, 'year': [m[:4] for m in s.index], 'mon': [int(m[5:7]) for m in s.index]})
    full = df.groupby('year')['mon'].transform('size') == 12
    df = df[full]
    df['ratio'] = df['v'] / df.groupby('year')['v'].transform('mean') * 100
    return df.groupby('mon')['ratio'].mean().rename('seasonal_index')


# ---------------------------------------------------------------- разбиение по времени
def _month_num(m):
    m = pd.Series(m).astype(str)
    return m.str[:4].astype(int) * 12 + m.str[5:7].astype(int) - 1


def time_split(df, train_end, test_start, gap=0, date_col='month'):
    """
    Делит таблицу по времени: train — строки с date_col ≤ train_end, test — ≥ test_start.
    gap — сколько месяцев должно лежать МЕЖДУ ними (для метки на 12 мес. вперёд нужно gap ≥ 12),
    иначе ошибка: так в train не попадут метки, посчитанные по тестовому периоду.
    """
    between = int(_month_num([test_start])[0] - _month_num([train_end])[0] - 1)
    if between < gap:
        raise ValueError(f'Между train_end={train_end} и test_start={test_start} {between} мес., нужно не меньше {gap}')
    d = df[date_col].astype(str)
    return df[d <= train_end].copy(), df[d >= test_start].copy()


def yoy_log(s, periods=12):
    """Рост год к году в логарифмах: log(x_t + 1) − log(x_{t−12} + 1). Убирает тренд и сезонность."""
    return np.log1p(pd.Series(s, dtype=float)).diff(periods)


# ---------------------------------------------------------------- календарь
def load_calendar(path=None):
    """Читает data/clean/calendar.csv (колонки — ТЗ 16.2)."""
    return pd.read_csv(path or clean_path('calendar.csv'), dtype={'month': str})


# ---------------------------------------------------------------- запросы
SYNONYMS = {   # ТЗ 16.4: складываем в один запрос
    'ануа': 'anua',
    'медикуб': 'medicube',
    'пдрн': 'pdrn',
    'санскрин': 'солнцезащитный крем',
    'витамин c': 'витамин с',   # латинская c (Naver) → кириллическая с (Вордстат)
}
DROP_TERMS = {'лосось днк'}        # почти пустой ряд
GROUP_EN = {'общий': 'general', 'бренд': 'brand', 'ингредиент': 'ingredient', 'категория': 'category'}


def normalize_term(term, group=None):
    """Приводит запрос к общему виду: нижний регистр, синонимы; у ингредиентов убирает « для лица»."""
    t = str(term).strip().lower()
    t = SYNONYMS.get(t, t)
    if group in ('ingredient', 'ингредиент') and t.endswith(' для лица'):
        t = t[: -len(' для лица')]
    return t
