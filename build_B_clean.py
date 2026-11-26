"""
build_B_clean.py — чистые файлы роли B (ТЗ v4, раздел 16.2).
Запуск из корня репозитория:  python build_B_clean.py
Пишет в data/clean/:
  calendar.csv          — календарь 2015-01 … 2026-12 (всем)
  search_clean.csv      — весь поиск в одном длинном формате (Лере и Гузель)
  keywords.csv          — словарь запросов (Лере)
  panel_search_export.csv — страна × месяц: поиск + экспорт в тоннах (вопрос 2)
Сырые файлы не меняются.
"""
import numpy as np
import pandas as pd
from utils import (load_expanded, zero_share, month_range, normalize_term, raw_path, clean_path,
                   GROUP_EN, DROP_TERMS, KR_TO_ISO2, yoy_log, LAST_MONTH, SEARCH_START)

log = []

# ================================================================ 1. calendar.csv
SEOLLAL = {2015: 2, 2016: 2, 2017: 1, 2018: 2, 2019: 2, 2020: 1, 2021: 2, 2022: 2, 2023: 1, 2024: 2, 2025: 1, 2026: 2}
CHUSEOK = {y: (10 if y in (2017, 2020, 2025) else 9) for y in range(2015, 2027)}
cal = pd.DataFrame({'month': month_range('2015-01', '2026-12')})
y, m = cal.month.str[:4].astype(int), cal.month.str[5:7].astype(int)
cal['m_dec'] = (m == 12).astype(int)                   # Новый год: подарки, распродажи
cal['m_mar'] = (m == 3).astype(int)                    # 8 Марта, Наурыз
cal['m_nov'] = (m == 11).astype(int)                   # 11.11 и «чёрная пятница»
cal['seollal'] = (m == y.map(SEOLLAL)).astype(int)     # корейский Новый год (ТЗ 16.4)
cal['chuseok'] = (m == y.map(CHUSEOK)).astype(int)     # Чусок (ТЗ 16.4)
cal['post_2022'] = (cal.month >= '2022-03').astype(int)      # санкции и уход западных брендов
cal['post_2025_10'] = (cal.month >= '2025-10').astype(int)   # маркировка косметики
cal.to_csv(clean_path('calendar.csv'), index=False)
log.append(f'calendar.csv: {cal.shape}')

# ================================================================ 2. search_clean.csv
parts = []
# 01: «корейская косметика» (T01 — 4 страны на одной шкале) и T04 — Казахстан + 4 бренда
g = load_expanded(raw_path('01_trends_generic_monthly_expanded.csv'), ['query_id', 'geo', 'term'])
g['group'] = np.where(g['term'] == 'корейская косметика', 'general', 'brand')
parts.append(g.assign(source='trends', export_id=g['query_id'], term=g['term'].map(normalize_term)))
# 02: корейские бренды одним запросом, с Medicube и без
b = load_expanded(raw_path('02_trends_brands_monthly_expanded.csv'), ['variant', 'geo'])
parts.append(b.assign(source='trends', export_id=b['variant'], term='k_brands', group='brand'))
# 03: 5 ингредиентов, шкала своя в каждой стране
i = load_expanded(raw_path('03_trends_ingredients_monthly_expanded.csv'), ['geo', 'ingredient'])
parts.append(i.assign(source='trends', export_id='ing', group='ingredient',
                      term=i['ingredient'].map(lambda t: normalize_term(t, 'ingredient'))))
# 05: Вордстат, Россия — абсолютные числа; синонимы складываем
w = pd.read_csv(raw_path('05_wordstat_ru_monthly.csv'), encoding='utf-8-sig')
w = w[(w['month'] >= SEARCH_START) & (w['month'] <= LAST_MONTH)]
w['group'] = w['group'].map(GROUP_EN)
w['term_raw'] = w['query']
w['term'] = [normalize_term(t, gr) for t, gr in zip(w['query'], w['group'])]
w = w[~w['term'].isin(DROP_TERMS)]
w = w.groupby(['geo', 'group', 'term', 'month'], as_index=False)['queries_per_month'].sum()
parts.append(w.rename(columns={'queries_per_month': 'value'}).assign(source='wordstat', export_id='ws'))

cols = ['source', 'export_id', 'geo', 'term', 'group', 'month', 'value']
search = pd.concat([p[cols] for p in parts], ignore_index=True)
key = ['source', 'export_id', 'geo', 'term']
search['zero_share'] = search.groupby(key)['value'].transform(zero_share).round(3)
search = search.sort_values(key + ['month']).reset_index(drop=True)
assert not search.duplicated(key + ['month']).any(), 'дубли ряд × месяц'
assert search['month'].max() <= LAST_MONTH and search['value'].notna().all()
assert search.groupby(key).size().eq(len(month_range())).all(), 'у ряда не 82 месяца'
search.to_csv(clean_path('search_clean.csv'), index=False)
n_series = search.groupby(key).ngroups
n_bad = search.groupby(key)['zero_share'].first().gt(0.5).sum()
log.append(f'search_clean.csv: {search.shape}, рядов {n_series}, из них нулей > 50%: {n_bad}')

# ================================================================ 3. keywords.csv
naver = None
try:
    naver = pd.read_csv(raw_path('naver_datalab_korea_monthly_2016-01_2026-10.csv'), encoding='utf-8-sig')
except FileNotFoundError:
    pass
KO_BRANDS = {'medicube': '메디큐브', 'anua': '아누아', 'round lab': '라운드랩', 'skin1004': '스킨1004',
             'cosrx': '코스알엑스', 'beauty of joseon': '조선미녀', 'missha': '미샤', 'torriden': '토리든'}
EN = {'корейская косметика': 'korean cosmetics', 'pdrn': 'pdrn', 'ниацинамид': 'niacinamide',
      'центелла': 'centella', 'ретинол': 'retinol', 'азелаиновая кислота': 'azelaic acid'}
KO = dict(KO_BRANDS)
if naver is not None:
    for t, en, ko in naver[['term', 'term_en', 'term_ko']].drop_duplicates().itertuples(index=False):
        t = normalize_term(t, 'ingredient')
        EN.setdefault(t, en); KO.setdefault(t, ko)

rows = []
w_raw = pd.read_csv(raw_path('05_wordstat_ru_monthly.csv'), encoding='utf-8-sig')[['query', 'group']].drop_duplicates()
for q, gr in w_raw.itertuples(index=False):
    gr = GROUP_EN[gr]
    rows.append((q, normalize_term(q, gr), gr, 'wordstat'))
rows += [('корейская косметика', 'корейская косметика', 'general', 'trends'),
         ('k_brands', 'k_brands', 'brand', 'trends')]
rows += [(t, t, 'brand', 'trends') for t in ['anua', 'beauty of joseon', 'cosrx', 'round lab']]
rows += [(t, normalize_term(t, 'ingredient'), 'ingredient', 'trends') for t in i['ingredient'].unique()]
kw = pd.DataFrame(rows, columns=['term_raw', 'term', 'group', 'source'])
kw = kw.groupby(['term_raw', 'term', 'group'], as_index=False)['source'].agg(lambda s: ', '.join(sorted(set(s))))
kw['k_specific'] = ((kw['group'] == 'brand') | (kw['term'] == 'корейская косметика')).astype(int)
kw['dropped'] = kw['term'].isin(DROP_TERMS).astype(int)
kw['term_en'] = kw['term'].map(lambda t: EN.get(t, t if t.isascii() else ''))
kw['term_ko'] = kw['term'].map(lambda t: KO.get(t, ''))
kw = kw[['term_raw', 'term', 'group', 'k_specific', 'term_en', 'term_ko', 'source', 'dropped']].sort_values(['group', 'term', 'term_raw'])
kw.to_csv(clean_path('keywords.csv'), index=False)
log.append(f'keywords.csv: {kw.shape}, синонимов слито: {(kw.term_raw != kw.term).sum()}, '
           f'с корейским названием: {(kw.term_ko != "").sum()}')

# ================================================================ 4. panel_search_export.csv (вопрос 2)
# Пока нет customs_clean.csv от Гузель — читаем таможню напрямую (тонны = 수출 중량), ТЗ 16.2.
customs_clean = clean_path('customs_clean.csv')
if customs_clean.exists():
    cu = pd.read_csv(customs_clean, dtype={'month': str}, encoding='utf-8-sig')[['month', 'iso2', 'tons', 'usd_k']]
    customs_src = 'customs_clean.csv'
else:
    c = pd.read_csv(raw_path('customs_HS3304_CIS_2015-01_2026-10.csv'), encoding='utf-8-sig')
    c['iso2'] = c['국가'].map(KR_TO_ISO2)
    cu = c.rename(columns={'기간': 'month', '수출 중량': 'tons', '수출 금액': 'usd_k'})[['month', 'iso2', 'tons', 'usd_k']]
    customs_src = 'customs_HS3304_CIS (сырой)'
GEOS = ['RU', 'KZ', 'BY', 'KG', 'UZ']
grid = pd.MultiIndex.from_product([GEOS, month_range('2019-01', LAST_MONTH)], names=['iso2', 'month']).to_frame(index=False)
cu = grid.merge(cu, on=['iso2', 'month'], how='left').fillna({'tons': 0, 'usd_k': 0})

def pick(df, **kw_):
    sel = df
    for k, v in kw_.items():
        sel = sel[sel[k] == v]
    return sel.rename(columns={'geo': 'iso2'})[['iso2', 'month', 'value']]

# поиск по стране: «корейская косметика» и бренды (без Medicube — основной)
gen = pd.concat([pick(search, source='trends', export_id='T01'),
                 pick(search, source='wordstat', term='корейская косметика')]).rename(columns={'value': 'search_general'})
br_wo = pd.concat([pick(search, export_id='without_medicube'),
                   search[(search.source == 'wordstat') & (search.group == 'brand') & (search.term != 'medicube')]
                   .groupby(['geo', 'month'], as_index=False)['value'].sum().rename(columns={'geo': 'iso2'})]
                  ).rename(columns={'value': 'search_brands'})
br_w = pd.concat([pick(search, export_id='with_medicube'),
                  search[(search.source == 'wordstat') & (search.group == 'brand')]
                  .groupby(['geo', 'month'], as_index=False)['value'].sum().rename(columns={'geo': 'iso2'})]
                 ).rename(columns={'value': 'search_brands_medicube'})
panel = cu.merge(gen, on=['iso2', 'month'], how='left').merge(br_wo, on=['iso2', 'month'], how='left') \
          .merge(br_w, on=['iso2', 'month'], how='left')
panel['search_source'] = np.where(panel['iso2'] == 'RU', 'wordstat', 'trends')
panel = panel.sort_values(['iso2', 'month']).reset_index(drop=True)
for col in ['tons', 'search_general', 'search_brands']:
    panel[f'{col}_yoy_log'] = panel.groupby('iso2')[col].transform(yoy_log)
for k in range(1, 7):
    for col in ['search_general_yoy_log', 'search_brands_yoy_log']:
        panel[f'{col}_lag{k}'] = panel.groupby('iso2')[col].shift(k)
panel = panel[panel['month'] >= SEARCH_START].reset_index(drop=True)   # 2019 нужен только для роста г/г
panel.to_csv(clean_path('panel_search_export.csv'), index=False)
log.append(f'panel_search_export.csv: {panel.shape}, таможня из: {customs_src}')

print('\n'.join(log))
(clean_path('B_build_log.txt')).write_text('\n'.join(log), encoding='utf-8')
