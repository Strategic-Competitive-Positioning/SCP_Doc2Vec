import sys
import os
import random
import pandas as pd
import datetime
import numpy as np
import scp_calculation as structhole
from functools import partial
random.seed(0)

def generate_threshold_similarity_matrix(similarity):
    '''similarity is calculated using cosine distance'''
    return np.where(similarity <= THRESHOLD, 1, 0)

def get_new_firm_index_metadata(index):
    k = next_year_indices[index]
    metadata = data00[k]['meta']
    return metadata

def get_new_firm_index_cik(index):
    metadata = get_new_firm_index_metadata(index)
    return metadata['CIK']

def get_new_firm_index_name(index):
    metadata = get_new_firm_index_metadata(index)
    return metadata['CONFORMED-NAME']

def convert_to_original_index(boolean_mask, index):
    '''return the original index by undoing the filtering of the boolean mask'''
    original_index = np.argwhere(boolean_mask == True).ravel()
    return original_index[index]

def get_subset_tril(boolean_mask):
    '''select subset of the similarity matrix using boolean mask then set values at and above the diagonal to 0'''
    subset = np.tril(threshold_similarity_current_year[boolean_mask,:][:,boolean_mask],k=-1)
    return subset

def set_triu_value(matrix, value):
    '''set the value of the upper triangle in the matrix to a given value'''
    triu_indicies = np.triu_indices_from(matrix)
    triu_matrix = matrix.copy()
    triu_matrix[triu_indicies] = value
    return triu_matrix

def count_num_non_hole_for_k(boolean_mask):
    '''count the number of 1s in the lower triangle of the subset'''
    subset_current_year = get_subset_tril(boolean_mask)
    return np.sum(np.sum(subset_current_year))
    
def get_sum_row_col_of_subset(subset, boolean_mask, count_fn):
    subset_with_triu_filled = set_triu_value(subset, -1)
    open_list = np.where(subset_with_triu_filled == 0)
    open_rows, open_cols = open_list
    v_convert_to_original_index = np.vectorize(partial(convert_to_original_index, boolean_mask))
    if len(open_rows) == 0:
        return 0
    open_rows = v_convert_to_original_index(open_rows)
    open_cols = v_convert_to_original_index(open_cols)
    open_coordinates = np.array([open_rows, open_cols]).T
    return count_fn(open_coordinates)

def get_hole(open_coordinate):
    open_row, open_col = open_coordinate
    connected_a = threshold_similarity_current_year[open_row,:]
    connected_b = threshold_similarity_current_year[:,open_col]
    hole_count = np.sum(np.logical_and(connected_a, connected_b.T))
    return hole_count

vget_holes = np.vectorize(get_hole, signature='(n)->()', otypes=[int])

def count_open_holes(open_coordinates):
    hole_count = vget_holes(open_coordinates)
    return np.sum(hole_count == 0)
    
def count_entered_holes(open_coordinates):
    hole_count = vget_holes(open_coordinates)
    return np.sum(hole_count != 0)

def count_num_hole_entering_for_k(boolean_mask):
    subset = get_subset_tril(boolean_mask)
    return get_sum_row_col_of_subset(subset, boolean_mask, count_entered_holes)

def count_num_hole_opening_for_k(boolean_mask):
    subset = get_subset_tril(boolean_mask)
    return get_sum_row_col_of_subset(subset, boolean_mask, count_open_holes)

def get_count_per_k(count_per_k):
    default_value = np.zeros(len(connected_to_at_least_two))
    default_value[np.where(connected_to_at_least_two == True)] = count_per_k
    return default_value.astype(int)

def get_df_count_per_k(count_per_k, label):
    index_k = np.arange(len(connected_to_at_least_two))
    count_dict = dict()
    count_dict['new_firm_index'] = index_k
    count_dict[label] = get_count_per_k(count_per_k)
    return pd.DataFrame.from_dict(count_dict)

for YEAR in range(1995,2021):
    # Steps 1-2
    tensor = pd.read_csv('tensor_{}_{}.tsv'.format(YEAR, YEAR+1), header=None, sep='\t')
    meta = pd.read_csv('corpus.raw.txt.{}-{}_metadata.tsv'.format(YEAR, YEAR+1), sep='\t')
    meta['SIC'] = meta['SIC'].astype(float)
    meta['YEAR'] = meta['YEAR'].astype(str).str[:-1].astype(int)
    tenkq = pd.read_csv('cik_year_10KQ.csv')
    data = dict(zip(tensor.index.values, [{'tensor': val} for val in tensor.values]))
    tensor_matrix = tensor.values
    for i, row in meta.iterrows():
        if i in data:
            data[i]['meta'] = row
    data00, tensor_matrix00, similarity00, dist_list, firm_count = structhole.analyze_by_year(meta, tensor, YEAR)
    cik_old, cik_new, cik00, cik01 = set(tenkq[tenkq.year <= YEAR].cik), set(tenkq[tenkq.year >= (YEAR+1)].cik), set(tenkq[tenkq.year == YEAR].cik), set(tenkq[tenkq.year == (YEAR+1)].cik)
    new01, s101 = set(cik01 - cik_old), set(s1df_t1['CIK'].values) 
    new01, s1prev = set(new01 | s101), set(s1df_prev['CIK'].values)
    new01_and_s1prev = set(new01 | s1prev)
    current_cik, next_cik, new_index01,current_year_index = set(), set(), set(), set()
    for i in data00:
        if data00[i]['meta']['YEAR'] != YEAR+1:
            continue
        cik = data00[i]['meta']['CIK']
        if cik not in new01_and_s1prev:
            continue
        if cik not in next_cik:
            new_index01.add(i)
            next_cik.add(cik)
    for i in data00:
        if data00[i]['meta']['YEAR'] == YEAR:
            cik = data00[i]['meta']['CIK']
            if cik not in current_cik:
                current_cik.add(cik)
                current_year_index.add(i)
    current_year_indices = list(current_year_index)
    next_year_indices = list(new_index01)
    similarity_current_year = similarity00[current_year_indices, :][:,current_year_indices]
    similarity_current_new_year = similarity00[current_year_indices, :][:,next_year_indices]
    threshold_similarity_current_year = generate_threshold_similarity_matrix(similarity_current_year) # threshold_similarity_current_year is Dt_binary

    # Steps 3-4
    threshold_similarity_current_next_year = generate_threshold_similarity_matrix(similarity_current_new_year)
    connected_to_at_least_two = np.sum(threshold_similarity_current_next_year, axis=0) > 1 # connected_to_at_least_two is S
    connected_to_at_least_two_matrix = threshold_similarity_current_next_year[:,connected_to_at_least_two] # connected_to_at_least_two_matrix is Dt*
    non_hole_count_per_k = np.apply_along_axis(count_num_non_hole_for_k, 1, (connected_to_at_least_two_matrix.T > 0))
    non_hole_count = sum(non_hole_count_per_k)

    # Steps 5-6
    hole_entering_count_per_k = np.apply_along_axis(count_num_hole_entering_for_k, 1, (connected_to_at_least_two_matrix.T > 0))
    hole_entering_count = sum(hole_entering_count_per_k)
    hole_opening_count_per_k = np.apply_along_axis(count_num_hole_opening_for_k, 1, (connected_to_at_least_two_matrix.T > 0))
    hole_opening_count = sum(hole_opening_count_per_k)
    non_hole_count_per_k_df = get_df_count_per_k(non_hole_count_per_k, 'non_holes_count_per_new_firm')
    hole_entering_count_per_k_df = get_df_count_per_k(hole_entering_count_per_k, 'hole_entering_count_per_new_firm')
    hole_opening_count_per_k_df = get_df_count_per_k(hole_opening_count_per_k, 'hole_opening_count_per_new_firm')

    df_results = pd.merge(pd.merge(non_hole_count_per_k, hole_entering_count_per_k_df, on='new_firm_index', how='left'), hole_opening_count_per_k_df, on='new_firm_index', how='left')
    df_results['CIK'] = df_results['new_firm_index'].map(get_new_firm_index_cik)
    df_results['name'] = df_results['new_firm_index'].map(get_new_firm_index_name)

    assert len(df_results['CIK'].unique()) == len(df_results)

    output_file = './output.csv'
    df_results.to_csv(output_file, index=False)
