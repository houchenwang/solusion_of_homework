import tensorcircuit as tc
import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tensorcircuit.cloud import apis

'''
function list:
to_binstr
dict_to_tensor
tensor_to_dict
normalize
trim_result
sample_local
gen_sample_cloud
calc_expectation
apply_matrix
get_confusion_matrix_single
get_confusion_matrix
apply_confusion_matrix
recalc_confusion_matrix
mitigated_sample
mitigated_expectation
'''

K = tc.set_backend("tensorflow")

# 将 x 转为长度为 len 的字符串，高位在前
def to_binstr(x: int, len: int) -> str:
    return "{1:0>{0}b}".format(len, x)

# 将字典 a 转化为长度为 2^n 的 tf.Tensor
def dict_to_tensor(n: int, a: dict):
    res = np.zeros(shape = (2 ** n), dtype = "complex64")
    for i in a:
        res[int(i, 2)] += a[i]
    return K.convert_to_tensor(res)

# 将长度为 2^n 的 tf.Tensor 转化为字典 a
def tensor_to_dict(n: int, a: tf.Tensor):
    res = {}
    for i in range(2 ** n):
        res[to_binstr(i, n)] = K.real(a[i])[0].numpy()
    return res

# 将字典 res 归一化
def normalize(res: dict):
    return tc.results.counts.normalized_count(res)

# 将字典 res 转化为 (list, 值) 构成的列表
def trim_result(res: dict):
    return [([int(x) for x in s], res[s]) for s in res]

# 对电路或电路列表测量 batch 次，返回理想测量结果（每种结果的次数）
def sample_local(cs, batch: int):
    if type(cs) == tc.Circuit:
        return cs.sample(batch = batch, allow_state = "True", format = "count_dict_bin")
    else:
        res = []
        for c in cs:
            res.append(c.sample(batch = batch, allow_state = "True", format = "count_dict_bin"))
        return res

# 给定 device_name，返回如下函数：
# 对电路或电路列表测量 batch 次，返回在 device_name 上的测量结果（每种结果的次数）
def gen_sample_cloud(device_name: str):
    def sample_cloud(cs, batch: int):
        t = apis.submit_task(provider="tencent",
            device=device_name,
            circuit=cs,
            shots=batch,
            enable_qos_qubit_mapping = False)
        if type(cs) == tc.Circuit:
            return t.results()
        else:
            res = []
            for x in t:
                res.append(x.results())
            return res
    return sample_cloud

# 给定字典形式的结果，返回 unitary 中各张量积的期望
def calc_expectation(result: dict, unitary: list[tf.Tensor]):
    
    p = trim_result(tc.results.counts.normalized_count(result))
    sum = 0
    for (id, x) in p:
        cur = 1
        for i in range(0, len(unitary)):
            cur *= unitary[i][id[i], id[i]]
        sum += cur * x
    return sum

# 给定字典和矩阵，将矩阵作用于字典
def apply_matrix(n: int, result: dict, mat: tf.Tensor):
    result = dict_to_tensor(n, result)
    result = mat @ K.reshape(result, shape = (2 ** n, 1))
    return tensor_to_dict(n, result)

# 返回给定比特列表 group 的 confusion_matrix
# n：比特数
# sample_function：测量函数
# batch：测量次数
# group：比特列表
def get_confusion_matrix_single(n: int, sample_function, batch: int, group: list[int]) -> tf.Tensor:
    
    d = len(group)
    cs = []
    for mask in range(2 ** d):
        c = tc.Circuit(n)
        maskstr = to_binstr(mask, d)
        for i in range(d):
            if(maskstr[i] == '1'):
                c.X(group[i])
        cs.append(c)
    
    res = sample_function(cs, batch)
    f = np.zeros(shape = (2 ** d, 2 ** d), dtype = "complex64")
    for mask in range(2 ** d):
        curres = res[mask]
        curres = tc.results.counts.normalized_count(curres)
        for s in curres:
            to = "".join(s[i] for i in group)
            f[mask][int(to, 2)] += curres[s]
    return K.convert_to_tensor(f)

# 返回全局的 confusion_matrix
# n：比特数
# sample_function：测量函数
# batch：测量次数
# group_list：比特的分组情况
def get_confusion_matrix(n: int, sample_function, batch: int, group_list: list[list[int]]) -> tf.Tensor:
    f = []
    for group in group_list:
        f.append(get_confusion_matrix_single(n, sample_function, batch, group))
    
    res = np.zeros(shape = (2 ** n, 2 ** n), dtype = "complex64")
    for i in range(2 ** n):
        for j in range(2 ** n):
            i_str = to_binstr(i, n)
            j_str = to_binstr(j, n)
            cur = 1
            for id in range(0, len(group_list)):
                group = group_list[id]
                x = "".join(i_str[k] for k in group)
                y = "".join(j_str[k] for k in group)
                cur *= f[id][int(x,2)][int(y,2)]
            res[i][j] = cur
    return K.convert_to_tensor(res)

# 作用某些位的 confusion_matrix
# res：初始结果
# idx：位的列表
# cmat：这些位的 confusion_matrix
def apply_confusion_matrix(res: dict, idx, cmat: tf.Tensor):
    if type(idx) == int:
        idx = [idx]
    
    cmat = K.inv(cmat)
    
    res_new = {}
    for s in res:
        sx = "".join(s[i] for i in idx)
        x = int(sx, 2)
        for y in range(0, 2 ** len(idx)):
            sy = to_binstr(y, len(idx))
            to = list(s)
            for i in range(0, len(idx)):
                to[idx[i]] = sy[i]
            to = "".join(to)
            if to in res:
                res_new.setdefault(to, 0)
                res_new[to] = res_new[to] + (K.real(res[s] * cmat[y][x]).numpy())
    return res_new

# 每个 device 每个比特的 confusion_matrix
all_confusion_matrix = {}

# 重新计算给定 device 每个比特的 confusion_matrix
def recalc_confusion_matrix(device_name: str, n: int):
    sample_cloud = gen_sample_cloud(device_name)
    cmat_list = []
    for i in range(0, n):
        cmat_list.append(get_confusion_matrix_single(n, sample_cloud, 8000, [i]))
    all_confusion_matrix[device_name] = cmat_list

# 测量电路，返回纠错后的结果
# device：设备名
# n：比特数
# batch：测试次数
# method: 使用误差消除的方法，目前有两种：confusion 或者 reduced
# - confusion 适用于所有电路，但在比特之间的误差不独立时不精确。需要 O(N^2) 空间。
# - reduced 不依赖于比特之间是否独立，但需要电路测出的结果种类数较少时才会快且精确，如果概率比较平均则会运行缓慢。空间开销大大减少。
# 
# limit_to_keep: 只会在使用 reduced 时用到，表示只接受出现次数 >= limit_to_keep 的结果作为测量对象。
# 其较大时速度快但结果不精确，较小时速度慢但结果更精确。
# allow_negative: 是否允许结果存在负数
# 返回元组 (纠错前, 纠错后)
def mitigated_sample(device: str, n: int, c: tc.Circuit, batch: int, method: str, limit_to_keep = None, allow_negative = False):
    if method == "confusion":
        if device not in all_confusion_matrix:
            recalc_confusion_matrix(device, n)
        
        res_origin = gen_sample_cloud(device)(c, batch)
        tmp = res_origin
        for i in range(n):
            tmp = apply_confusion_matrix(tmp, [i], all_confusion_matrix[device][i])
        
        if allow_negative:
            res = tmp
        else:
            res = {}
            for s in tmp:
                if tmp[s] >= 0:
                    res[s] = tmp[s]
        
        res_origin = normalize(res_origin)
        res = normalize(res)
        
        return res_origin, res 
    elif method == "reduced":
        res = gen_sample_cloud(device)([c], batch)[0]
        dic = {}
        tot = 0
        cs = []
        for st in res:
            if res[st] < limit_to_keep:
                continue
            dic[st] = tot
            d = tc.Circuit(n)
            for i in range(n):
                if st[i] == '1':
                    d.x(i)
            cs.append(d)
            tot += 1
        confusion_res = gen_sample_cloud(device)(cs, batch)
        f = np.zeros(shape = (tot, tot), dtype = "float64")
        for i in range(tot):
            for j in confusion_res[i]:
                if (j in dic) == False:
                    confusion_res[i][j] = 0
            confusion_res[i] = tc.results.counts.normalized_count(confusion_res[i])
            for j in confusion_res[i]:
                if j in dic:
                    f[i][dic[j]] = confusion_res[i][j]
        f = K.inv(K.convert_to_tensor(f))
        final_res = {}
        for i in dic:
            final_res[i] = 0
        for i in dic:
            for j in dic:
                final_res[j] += res[i] * f[dic[j]][dic[i]]
        
        if not allow_negative:
            for i in dic:
                if final_res[i] < 0:
                    final_res[i] = 0
        
        for i in final_res:
            final_res[i] = K.real(final_res[i]).numpy()
        
        res = normalize(res)
        final_res = normalize(final_res)
        return res, final_res
    else:
        raise RuntimeError('Invalid method name.')

# 返回纠正误差后的期望值，目前只允许 x,z
# 前面的参数与 mitigated_sample 相同
# 特别地，method == "confusion direct" 表示利用 confusion_matrix 直接计算期望
# x: 测量 X 的比特列表
# z: 测量 Z 的比特列表
# 其它比特默认为 I
# 注意此函数会改变 c
# 返回元组 (纠错前, 纠错后)
def mitigated_expectation(device: str, n: int, c: tc.Circuit, batch: int, method: str, limit_to_keep = None, x: list = [], z: list = []):
    I = [[1, 0], [0, 1]]
    Z = [[1, 0], [0, -1]]
    
    unitary_list = [I] * n
    for i in x:
        unitary_list[i] = Z
    for i in z:
        unitary_list[i] = Z
    
    for i in x:
        c.H(i)
    
    if method == "confusion direct":
        p0, p1 = mitigated_sample(device, n, c, batch, "confusion", limit_to_keep, allow_negative = True)
        p1 = p0
        cmat_list = []
        for i in range(n):
            cmat_list.append(K.real(K.inv(all_confusion_matrix[device][i])).numpy())
    else:
        p0, p1 = mitigated_sample(device, n, c, batch, method, limit_to_keep, allow_negative = True)
        cmat_list = [I] * n
    
    res0 = 0
    for s in p0:
        cur = p0[s]
        for i in range(n):
            t = int(s[i])
            cur *= unitary_list[i][t][t]
        res0 += cur
    
    res1 = 0
    for s in p1:
        cur = p1[s]
        for i in range(n):
            t = int(s[i])
            cur *= unitary_list[i][0][0] * cmat_list[i][0][t] \
                +  unitary_list[i][1][1] * cmat_list[i][1][t]
        res1 += cur
    
    return res0, res1