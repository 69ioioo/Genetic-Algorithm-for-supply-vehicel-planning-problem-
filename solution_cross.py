import numpy as np
from numpy.testing._private.utils import rand
import pandas as pd
import math
import random
import copy
import itertools

def Check_frequency_match(check_list, cadidate_node):
    raw_checklist = list(map(int, np.hstack(np.array(check_list))))
    freq_list = []
    for i in range(len(Inst_indices)):
        freq_list.append([raw_checklist.count(i)])
    if freq_list[cadidate_node][0] < fq[cadidate_node]:
        return True
    else:
        return False
#將list做groupby & merge 的tool for check day t 是否有航班到 installation i use!
def List_GroupbyAndMerge(list_needGM, groupby_TGT):
    list_groupby = [[list[2] for list in list_needGM if list[0] == d] for d in groupby_TGT]
    list_groupby_merge = []
    for i in range(len(list_groupby)):
        list_groupby_merge.append([x for j in list_groupby[i] for x in j])
    return list_groupby_merge

def Installation_Distance(x, y):
    distance_matrix = []
    for i in range(len(x)):
        for j in range(len(y)):
            dist = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            distance_matrix.append(dist)
    distance_matrix = np.array(distance_matrix).reshape(len(x), len(y)).tolist()
    return distance_matrix

def Fitness_function(candidate_list): # candidata_list = s_new
    cost_list = []
    for i in range(len(candidate_list)):
        if len(candidate_list[2]) > 0:
            for j in range(len(candidate_list[i][2])-1):
                cost = distance_matrix[candidate_list[i][2][j]][candidate_list[i][2][j+1]]
                cost_list.append(cost)
    fitness = sum(cost_list)
    return fitness

def Cheapest_Installation(missing_list):
    # 欄位名稱 [欲插入點, 插入成本, 哪條路線, 路線中的插入位置, 前一個點位, 後一個點位]
    cheapest_series = []
    for i in missing_list:
        for j in range(len(s_new)):
            for k in range(len(s_new[j][2])):
                if k == 0:
                    cost = distance_matrix[k][i] + distance_matrix[s_new[j][2][k]][i] - distance_matrix[k][s_new[j][2][k]]
                    insert_pos = k
                    cheapest_series.append([i, cost, j, insert_pos, 0, s_new[j][2][k]])
                if k == len(s_new[j][2])-1:
                    cost_previous = distance_matrix[s_new[j][2][k-1]][i] + distance_matrix[s_new[j][2][k]][i] - distance_matrix[s_new[j][2][k-1]][s_new[j][2][k]]
                    insert_pos_pre = k
                    cheapest_series.append([i, cost_previous, j, insert_pos_pre, s_new[j][2][k-1], s_new[j][2][k]])
                    cost_depot = distance_matrix[s_new[j][2][k]][i] + distance_matrix[0][i] - distance_matrix[0][s_new[j][2][k]]
                    insert_pos_depot = k+1
                    cheapest_series.append([i, cost_depot, j, insert_pos_depot, s_new[j][2][k], 0])
                elif k != 0 and k != len(s_new[j][2])-1:
                    cost = distance_matrix[s_new[j][2][k-1]][i] + distance_matrix[s_new[j][2][k]][i] - distance_matrix[s_new[j][2][k-1]][s_new[j][2][k]]
                    insert_pos = k
                    cheapest_series.append([i, cost, j, insert_pos, s_new[j][2][k-1], s_new[j][2][k]])
    return cheapest_series
def Check_MissingInst(check_list):
    raw_checklist = list(map(int, np.hstack(np.array(check_list))))
    freq_list = []
    for i in range(len(Inst_indices)):
        freq_list.append([raw_checklist.count(i)])
    missing_list = []
    for i in range(len(Inst_indices)):
        if freq_list[i][0] < fq[i]:
            missing_list.append(i)
    if len(missing_list) > 0:
        list(np.hstack(np.array(missing_list)))
    else:
        missing_list = missing_list
    return missing_list

CENTER = 0 #supply depot
HUGE = 999999
#------------------------ Input data -------------------------------
number_of_installations = 30
dir = r'/Users/amoslee/Desktop/python/PVRP/data/{}P.csv'.format((number_of_installations))
df = pd.DataFrame(pd.read_csv(dir))
Inst_indices = list(df['n']) #installation編號1-30 包含depot: 0
X, Y = list(df['x']), list(df['y'])
distance_matrix = Installation_Distance(X, Y)
fq = list(df['f']) # 各installation補給頻率需求
Iq = list(df['nN']) # 各insallation補給需求量
n = len(df['n']) - 1 # installation數量
v = 5 # vehicle數量上限
vehicle_cap = 10 # vehicle容量上限
week_day = [1, 2, 3, 4, 5, 6, 7]
vehicle_number = np.arange(1, v+1).tolist()
depot_capacity = 2
service_gap_days = 1 #PSV每次服務與服務之間的間隔

#------------------------ GA控制參數 -------------------------------
popsize = 100 # population size
elite_propotion = 0.3 # Proportion of elite individuals
Rc = 0.7 # Rate of crossover = crossover offspring 在populatoin裡面的比例
Pm = 0.05 # Mutation probability
'''
------------------------ Individual Generator -------------------------------
'''
iteration = 0
Population = []
individuals_fitness =[]
while iteration <= 2:
    #------------------------ STEP 1: 生成初始 Installation Service Pattern -------------------------------
    init_InstallationPattern = []
    for Inst_index in range(1, len(Inst_indices)):
        init_InstallationPattern.append(sorted(np.random.choice(week_day, fq[Inst_index], replace = False)))
    Days_InstNeedService = np.unique(np.hstack(init_InstallationPattern)).tolist()

    #------------------------ STEP 2: 生成初始 Vessel Service Pattern -------------------------------
    # 限制(1): 每個psv pattern至少必須包含installation需要被服務天集合中的一天
    # 限制(2): Depot capacity不能被違反
    serviceList_day = []
    for i in week_day:
        temp_day = []
        for j in range(len(init_InstallationPattern)):
            if i in init_InstallationPattern[j]:
                temp_day.append(j+1)
        serviceList_day.append(temp_day)
    
    PSVservice_day = [[]]*len(week_day)
    while True:
        day_count = 1
        while day_count < len(week_day)+1:
            day_demandPSV = int(len(serviceList_day[day_count-1]) // vehicle_cap) + 1
            if day_count == 1:
                while True:
                    Psv_pattern = sorted(np.random.choice(vehicle_number, np.random.randint(day_demandPSV, depot_capacity+1), replace = False))
                    if len(set(Psv_pattern)&set(PSVservice_day[day_count])) == 0:  
                        PSVservice_day[day_count-1] = Psv_pattern
                        day_count += 1
                        break
            elif day_count != len(week_day) and day_count != 1:
                while True:
                    Psv_pattern = sorted(np.random.choice(vehicle_number, np.random.randint(day_demandPSV, depot_capacity+1), replace = False))
                    #if len(set(Psv_pattern)&set(PSVservice_day[day_count])) == 0:
                    if len(set(Psv_pattern)&set(PSVservice_day[day_count-2])) == 0:
                        PSVservice_day[day_count-1] = Psv_pattern
                        day_count += 1
                        break
            else:
                while True:
                    Psv_pattern = sorted(np.random.choice(vehicle_number, np.random.randint(day_demandPSV, depot_capacity+1), replace = False))
                    if len(set(Psv_pattern)&set(PSVservice_day[0])) == 0:
                        if len(set(Psv_pattern)&set(PSVservice_day[day_count-2])) == 0:
                            PSVservice_day[day_count-1] = Psv_pattern
                            day_count += 1
                            break
        #確保installations需要被服務的天數都有被包含＆ Depot capacity沒有被違反
        check_day = list(map(lambda x: len(x)>0, PSVservice_day))
        if all(check_day) == True:
            break
        else:
            del PSVservice_day[:]

    #------------------------ STEP 3: 組合Installation Service Pattern & PSV pattern 生成 Tour chromosome -------------------------------
    Individual = []
    for i in range(len(PSVservice_day)):
        if len(PSVservice_day[i]) == 1:
            for j in range(len(vehicle_number)):
                if PSVservice_day[i][0] == j+1:
                    Individual.append([i+1, PSVservice_day[i][0], serviceList_day[i]])
                else:
                    Individual.append([i+1, j+1, []])
        else:
            #random.shuffle(serviceList_day[i])
            temp_serviceList_day = copy.deepcopy(serviceList_day[i])
            _sum = len(serviceList_day[i])
            n = len(PSVservice_day[i])
            while True:
                slic_temp = np.random.multinomial(_sum, np.ones(n)/n).tolist()
                if all(list(map(lambda x: x <= vehicle_cap, slic_temp))) == True:
                    slic = slic_temp
                    break
            for j in range(len(vehicle_number)):
                for k in range(n):
                    if PSVservice_day[i][k] == j+1:
                        #for l in range(len(slic)):
                        rand_choice = np.random.choice(temp_serviceList_day, slic[k], replace=False).tolist()
                        set_rand_choice = set(rand_choice)
                        set_mom = set(temp_serviceList_day)
                        temp_serviceList_day = list(set_mom.difference(set_rand_choice))
                        Individual.append([i+1, PSVservice_day[i][k], rand_choice])
                if j+1 not in PSVservice_day[i]:
                    Individual.append([i+1, j+1, []])
    iteration += 1
    Population.append(Individual)
    in_fit = Fitness_function(Individual)
    individuals_fitness.append([Population.index(Individual),in_fit])
individuals_fitness.sort(key = lambda x: x[1])
# 定義 s1 & s2 (Parents)
s1, s2 = Population[0], Population[1]
'''
------------------------ Crossover Operator-------------------------------
'''
# 從0到35(總規劃天數*總PSV數量)隨機選二數字，小者為n1大者為n2
Tot_couples = len(week_day) * len(vehicle_number)
Randchoose_uniform = sorted(np.random.choice(range(1, Tot_couples), 2, replace = False))
n1, n2 = Randchoose_uniform[0], Randchoose_uniform[1]
# 定義lambda_1, lambda_2, lambda_mix
lambda_1, lambda_2, lambda_mix = int(n1), int(n2-n1), int(Tot_couples-n2)
s_new = []

#------------------------ Crossover procedure STEP 1: copy all elements from s1 data -------------------------------
temp_s1 = copy.deepcopy(s1)
random.shuffle(temp_s1)
for i in range(lambda_1):
    s_new.append(temp_s1[i])
del temp_s1[:lambda_1]
#------------------------ Crossover procedure STEP 2: copy installation sequence between cut points from lambda mix data -------------------------------
temp_lambda_mix = []
for i in range(lambda_mix):
    temp_lambda_mix.append(temp_s1[i])
cutpoint = np.random.choice(range(0, len(temp_lambda_mix)), 2, replace = False)
alph_1, alph_2 = cutpoint[0], cutpoint[1] #alph_1: inclusive & alph_2: exclusive
if alph_1 < alph_2:
    for j in range(alph_1, alph_2):
        s_new.append(temp_lambda_mix[j])
else:
    del temp_lambda_mix[alph_2:alph_1]
    for j in range(len(temp_lambda_mix)):
        s_new.append(temp_lambda_mix[j])
s_new = sorted(list(filter(lambda x: any(x[2])==True, s_new)))

#------------------------ Crossover procedure STEP 3: copy elements from s2 data -------------------------------
# temp_s1裡面裝的就是 lambda_2 & lambda_mix的聯集
temp_s2 = copy.deepcopy(s2)
union_list = [] #lambda_2 跟 lambda_mix 聯集的list
for i in range(len(temp_s2)):
    for j in range(len(temp_s1)):
        if temp_s2[i][0] == temp_s1[j][0] and temp_s2[i][1] == temp_s1[j][1]:
            union_list.append(temp_s2[i])

# union_list_GM = List_GroupbyAndMerge(union_list, week_day)
s_new_GM = List_GroupbyAndMerge(s_new, week_day)
voyage_sum = [[list[1] for list in s_new if list[0] == d and len(list[2])!=0] for d in week_day]

    #------------------------ 檢查 day t 是否有航班到 installation i ?? -------------------------------
for i in range(len(s_new_GM)):
    for j in range(len(union_list)): 
        if union_list[j][0] == i+1 and len(union_list[j][2]) != 0:
            for k in range(len(union_list[j][2])):
                if union_list[j][2][k] not in s_new_GM[i]: # 檢查 day t 是否有航班到 installation i 
                    #------------------------ 從union list檢查PSV v 在day t 是否有出航？ -------------------------------
                    #------------------------ PSV v 在day t 有出航
                    if union_list[j][1] in voyage_sum[i]:
                        for l in range(len(s_new)):
                            if s_new[l][0] == i+1 and s_new[l][1] == union_list[j][1] and Check_frequency_match(s_new_GM, union_list[j][2][k]) == True:
                                # PSV v 在day t 有出航 且 限制 vehicle capacity滿了就不能插入new node
                                if len(s_new[l][2]) < vehicle_cap:
                                    s_new[l][2].append(union_list[j][2][k])
                                    s_new_GM = List_GroupbyAndMerge(s_new, week_day)
                                    
                    #------------------------PSV v 在day t 無出航 且無違反depot_capacity
                    elif union_list[j][1] not in voyage_sum[i] and len(voyage_sum[i]) < depot_capacity:
                        if Check_frequency_match(s_new_GM, union_list[j][2][k]) == True:
                            # 加入PSV每次服務與服務之間的間隔 至少一天限制
                            if i == len(voyage_sum)-1: # i==Sunday 要查monday是否有航班
                                if union_list[j][1] not in voyage_sum[i-1]:
                                    if union_list[j][1] not in voyage_sum[0]:
                                        s_new.append([i+1, union_list[j][1], [union_list[j][2][k]]])
                                        voyage_sum[i].append(union_list[j][1])
                                        s_new_GM = List_GroupbyAndMerge(s_new, week_day)
                            elif i == 0: # i==Monday 要查 Sunday是否有航班
                                if union_list[j][1] not in voyage_sum[i+1]:
                                    if union_list[j][1] not in voyage_sum[len(voyage_sum)-1]:
                                        s_new.append([i+1, union_list[j][1], [union_list[j][2][k]]])
                                        voyage_sum[i].append(union_list[j][1])
                                        s_new_GM = List_GroupbyAndMerge(s_new, week_day)
                            else:
                                if union_list[j][1] not in voyage_sum[i-1]:
                                    if union_list[j][1] not in voyage_sum[i+1]:
                                        s_new.append([i+1, union_list[j][1], [union_list[j][2][k]]])
                                        voyage_sum[i].append(union_list[j][1])
                                        s_new_GM = List_GroupbyAndMerge(s_new, week_day)

#------------------------ Crossover procedure STEP 4: Insert missing installations -------------------------------
missing_installations = Check_MissingInst(s_new_GM)
while True:
    insert_candidates = Cheapest_Installation(missing_installations)
    # 根據成本排序並修剪掉cost==0（代表插入點跟原本的相同）的候選點位
    insert_candidates_trim = sorted(list(filter(lambda x: x[1]>0, insert_candidates)), key = lambda y: y[1])
    # 欄位名稱 [欲插入點, 插入成本, 哪條路線, 路線中的插入位置, 前一個點位, 後一個點位]
    # insert_confirminstallaion = insert_candidates_trim[0]
    for i in range(len(insert_candidates_trim)): # 從最小成本的插入點開始插入，若最小的插入點vehicle cap已滿則選擇次小者，以此類推
        if len(s_new[insert_candidates_trim[i][2]][2]) < vehicle_cap:
            s_new[insert_candidates_trim[i][2]][2].insert(insert_candidates_trim[i][3], insert_candidates_trim[i][0])
            s_new_GM = List_GroupbyAndMerge(s_new, week_day)
            missing_installations = Check_MissingInst(s_new_GM)
            break
    if len(missing_installations) == 0:
        break
'''
------------------------ Mutation Operators (6 differnet kinds)-------------------------------
'''
def Mutation_operators(candidata_individual):
    dice = np.random.randint(1,7)
    # def Exchange_Mutation(candidata_individual): # 路徑中任選兩點交換位置
    if dice == 1:
        muta_points = np.random.choice(range(0, len(candidata_individual)-1), 2, replace=False)
        change_x, change_y = muta_points[0], muta_points[1]
        candidata_individual[change_x], candidata_individual[change_y] = candidata_individual[change_y], candidata_individual[change_x]
    # def Scramble_Mutation(candidata_individual): # 路徑中任選一段點位並進行段內重新隨機排序
    if dice == 2:
        while True:
            subset_len = np.random.randint(2, vehicle_cap+1)
            start_point = np.random.choice(range(len(candidata_individual)), 1, replace = False)
            sub = candidata_individual[start_point[0] : start_point[0]+subset_len]
            random.shuffle(sub)
            if len(sub) > 1:
                candidata_individual[start_point[0] : start_point[0]+subset_len] = sub
                break
    # def Displacement_Mutation(candidata_individual): # 路徑中任選一段點位並換位置
    if dice == 3:
        while True:
            subset_len = np.random.randint(2, vehicle_cap+1)
            start_point = np.random.choice(range(len(candidata_individual)), 2, replace = False)
            sub = candidata_individual[start_point[0] : start_point[0]+subset_len]
            insert_pos = start_point[1]
            if len(sub) > 1:
                del candidata_individual[start_point[0] : start_point[0]+subset_len]
                candidata_individual[insert_pos : insert_pos] = sub
                break
    # def Insertion_Mutation(candidata_individual): # 路徑中任選一點並換位置插入
    if dice == 4:    
        choose_point = np.random.choice(range(len(candidata_individual)), 2, replace = False)
        muta_point = candidata_individual[choose_point[0]]
        insert_pos = choose_point[1]
        del candidata_individual[choose_point[0]]
        candidata_individual.insert(insert_pos, muta_point)
    # def Inversion_Mutation(candidata_individual): # 路徑中任選一段點位並倒置順序
    if dice == 5:    
        while True:
            subset_len = np.random.randint(2, vehicle_cap+1)
            start_point = np.random.choice(range(len(candidata_individual)), 1, replace = False)
            sub = candidata_individual[start_point[0] : start_point[0]+subset_len]
            sub.reverse()
            if len(sub) > 1:
                candidata_individual[start_point[0] : start_point[0]+subset_len] = sub
                break
    # def DisplacedInversion_Mutation(candidata_individual): # 路徑中任選一段點位更換位置並且倒置順序
    if dice == 6:   
        while True:
            subset_len = np.random.randint(2, vehicle_cap+1)
            start_point = np.random.choice(range(len(candidata_individual)), 2, replace = False)
            sub = candidata_individual[start_point[0] : start_point[0]+subset_len]
            sub.reverse()
            insert_pos = start_point[1]
            if len(sub) > 1:
                del candidata_individual[start_point[0] : start_point[0]+subset_len]
                candidata_individual[insert_pos : insert_pos] = sub
                break
    return candidata_individual
# mutation operator is chosen randomly from the above six types
check_prob = random.uniform(0, 1)
if check_prob < Pm:
    s_new_readyforMutation = List_GroupbyAndMerge(s_new, week_day)
    s_new_AfterMutation = Mutation_operators(s_new_readyforMutation)

# 在每條route前後加上depot index "0"
for i in range(len(s_new)):
    s_new[i][2].insert(0, 0)
    s_new[i][2].append(0)

fit = Fitness_function(s_new)
# import os
# import csv
p=0