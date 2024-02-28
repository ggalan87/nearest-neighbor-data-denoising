import pickle


def cm_to_f1(cm, n_noisy, n_clean):
    true_noisy, false_noisy = cm[0] * n_noisy
    false_clean, true_clean = cm[1] * n_clean

    tp, fp = true_clean, false_clean
    tn, fn = true_noisy, false_noisy

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


def read_openset():
    with open('./configs_and_results_openset_cars.pkl', 'rb') as f:
        data_all = pickle.load(f)

    max_f1_score = -1
    max_idx = -1

    n_noisy = 4012
    n_clean = 4020
    for i, entry in enumerate(data_all):
        cm = entry['confusion_matrix']

        f1 = cm_to_f1(cm, n_noisy, n_clean)
        if f1 > max_f1_score:
            max_f1_score = f1
            max_idx = i

    print(max_f1_score, data_all[max_idx])


def read_knn():
    with open('./configs_and_results_knn_cars.pkl', 'rb') as f:
        data_all = pickle.load(f)

    max_f1_score = -1
    max_idx = -1

    n_noisy = 4012
    n_clean = 4020
    for i, entry in enumerate(data_all):
        cm = entry['confusion_matrix']

        f1 = cm_to_f1(cm, n_noisy, n_clean)
        if f1 > max_f1_score:
            max_f1_score = f1
            max_idx = i

    print(max_f1_score, data_all[max_idx])


read_openset()
read_knn()
