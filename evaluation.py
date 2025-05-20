# evaluation.py

from surprise import accuracy
from collections import defaultdict

def evaluate_collaborative(model):
    """Evaluate collaborative model using RMSE and MAE."""
    predictions = model.model.test(model.testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    return rmse, mae

def get_top_n(predictions, n=10):
    """Return the top-N recommendations for each user from a list of predictions."""
    top_n = defaultdict(list)

    for pred in predictions:
        uid = pred.uid
        iid = pred.iid
        est = pred.est
        top_n[uid].append((iid, est))

    # Sort predictions for each user and retrieve the top N highest ones
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def evaluate_precision_recall_f1(model, threshold=3.5, top_n=10):
    """Evaluate model using Precision, Recall, and F1-score at top N."""
    predictions = model.model.test(model.testset)
    topn_preds = get_top_n(predictions, n=top_n)

    precision_list = []
    recall_list = []

    for uid, user_ratings in topn_preds.items():
        # model.testset contains (user, item, true_rating)
        n_rel = sum((true_r >= threshold) for (user, item, true_r) in model.testset if user == uid)
        n_rec_k = len(user_ratings)
        n_rel_and_rec_k = sum((est >= threshold) for (_, est) in user_ratings)

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recall = n_rel_and_rec_k / n_rel if n_rel else 0

        precision_list.append(precision)
        recall_list.append(recall)

    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return avg_precision, avg_recall, f1