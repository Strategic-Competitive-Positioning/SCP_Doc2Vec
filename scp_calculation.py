import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy import sparse

def analyze_by_year(meta, tensor, year, sic):
    data, map_index, industry_min_max = dict(),dict(),dict()
    data_count = 0
    industry_min_max['agri-forest-fish'], industry_min_max['mining'], industry_min_max['construction'], industry_min_max['manufacturing'], industry_min_max['wholesale'], industry_min_max['retail'], industry_min_max['pubadmin'] = [100, 999], [1000, 1499], [1500, 1799], [2000, 3999], [5000, 5199], [5200, 5999], [9100, 9999]
    if not sic.startswith('ALL') and len(str(sic)) != 1:
        pass
    else:
        num, ff_map = 0, dict()
    for i, row in meta.iterrows():
        if row['YEAR'] != year and row['YEAR'] != year+1:
            continue
        try:
            this_sic = int(row['SIC'])
        except:
            continue
        if len(sic) == 1: 
            target_sic = int(sic)
            if this_sic < target_sic*1000 or this_sic > (target_sic+1)*1000:
                continue
        elif sic == 'tech':
            if (this_sic >= 3570 and this_sic <= 3577) or (this_sic == 3579) or (this_sic == 3600) or (this_sic >= 3661 and this_sic <= 3674) or (this_sic == 3845) or (this_sic >= 4812 and this_sic <= 4899) or (this_sic == 5045) or (this_sic == 5064) or (this_sic == 5065) or (this_sic >= 5731 and this_sic <= 5734) or (this_sic >= 7370 and this_sic <= 7377) or (this_sic == 7385):
                to_include = True
            else:
                to_include = False
            if to_include == False:
                continue
        elif sic == 'ALL-69':
            if this_sic >= 6000 and this_sic < 7000:
                continue
            if this_sic >= 9000:
                continue
        elif sic.startswith('SAMPLE_'):
            if random.random() > float(sic[7:]):
                continue
        elif sic in industry_min_max: 
            this_min, this_max= industry_min_max[sic][0], industry_min_max[sic][1]
            if this_sic < this_min or this_sic > this_max:
                continue
        elif sic == 'New_SIC2':
            if this_sic < 2000 or this_sic >= 4000:
                continue
        elif sic == 'New_SIC7':
            if this_sic < 7000 or this_sic >= 9000:
                continue
        data[data_count] = dict()
        data[data_count]['meta'] = row
        map_index[i] = data_count
        data_count += 1


    tensor_matrix = []
    for i, row in tensor.iterrows():
        if i not in map_index:
            continue
        data[map_index[i]]['tensor'] = row.values
        tensor_matrix.append(row.tolist())
    tensor_matrix = np.matrix(tensor_matrix)       
    tensor_sparse = sparse.csr_matrix(tensor_matrix)
    distances = cosine_distances(tensor_sparse)
    dist_list = []
    firm_count = 0
    for i in data:
        if data[i]['meta']['YEAR'] != year:
            continue
        firm_count += 1
        for j in data:
            if i >= j:
                continue
            if data[j]['meta']['YEAR'] != year:
                continue
            try:
                dist_list.append(distances[i][j])
            except:
                continue                
    return data, tensor_matrix, distances, dist_list, firm_count


