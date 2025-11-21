#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Display all extracted categories"""
import pandas as pd

df = pd.read_csv('config/category_mapping.csv')
print(f'Total categories extracted: {len(df)}\n')
print('All categories:\n')
for i, cat in enumerate(df['original_category'], 1):
    print(f'{i:3}. {cat}')
