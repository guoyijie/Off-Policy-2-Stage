import numpy as np
import os

one_precision5 = []
one_precision10 = []
one_recall5 = []
one_recall10 = []
one_ndcg5 = []
one_ndcg10 = []

two_precision5 = []
two_precision10 = []
two_recall5 = []
two_recall10 = []
two_ndcg5 = []
two_ndcg10 = []

for seed in range(11):
    #path='results/kl_weight_10.0_decay_0.9_s%d.npy'%seed
    path='results/ips_s%d.npy'%seed
    if not os.path.exists(path):
        break
    print(path)
    tmp = np.load(path, allow_pickle=True)
    tmp = tmp[()]
    best_epoch = tmp[0]
    test_results = tmp[-1]
    print('best epoch', best_epoch)
    one_test_result = test_results[best_epoch][0]
    two_test_result = test_results[best_epoch][1]
    one_precision5.append(one_test_result['Precision@5'])
    one_precision10.append(one_test_result['Precision@10'])
    one_recall5.append(one_test_result['Recall@5']) 
    one_recall10.append(one_test_result['Recall@10']) 
    one_ndcg5.append(one_test_result['NDCG@5']) 
    one_ndcg10.append(one_test_result['NDCG@10'])
    two_precision5.append(two_test_result['Precision@5'])
    two_precision10.append(two_test_result['Precision@10'])
    two_recall5.append(two_test_result['Recall@5'])
    two_recall10.append(two_test_result['Recall@10']) 
    two_ndcg5.append(two_test_result['NDCG@5'])     
    two_ndcg10.append(two_test_result['NDCG@10'])
print('one stage Precision@5', np.mean(one_precision5), np.std(one_precision5))
print('one stage Precision@10', np.mean(one_precision10), np.std(one_precision10))
print('one stage Recall@5', np.mean(one_recall5), np.std(one_recall5))
print('one stage Recall@10', np.mean(one_recall10), np.std(one_recall10))
print('one stage NDCG@5', np.mean(one_ndcg5), np.std(one_ndcg5))
print('one stage NDCG@10', np.mean(one_ndcg10), np.std(one_ndcg10))

print('two stage Precision@5', np.mean(two_precision5), np.std(two_precision5))
print('two stage Precision@10', np.mean(two_precision10), np.std(two_precision10))
print('two stage Recall@5', np.mean(two_recall5), np.std(two_recall5))
print('two stage Recall@10', np.mean(two_recall10), np.std(two_recall10))
print('two stage NDCG@5', np.mean(two_ndcg5), np.std(two_ndcg5))
print('two stage NDCG@10', np.mean(two_ndcg10), np.std(two_ndcg10))
