import numpy as np


def get_acc(phi_ik, truthfile):
    score = (phi_ik == phi_ik.max(axis=1, keepdims=True)).astype(np.float)
    score /= score.sum(axis=1, keepdims=True)
    return score[truthfile.item.values, truthfile.truth.values].sum() / truthfile.shape[0]


def get_fscore(predictions, truthfile):
    answers = {}
    for i in range(len(predictions)):
        list = predictions[i].tolist()
        answers[i] = list.index(max(list))

    def get_precision(truthfile, answers):
        items = truthfile['item'].tolist()
        truths = truthfile['truth'].tolist()
        truthdict = dict(zip(items, truths))
        total_sum = 0
        correct_sum = 0
        for item in answers.keys():
            if item in truthdict.keys():
                if answers[item] == 1:
                    total_sum += 1
                    if answers[item] == truthdict[item]:
                        correct_sum += 1
        return correct_sum / total_sum

    def get_recall(truthfile, answers):
        items = truthfile['item'].tolist()
        truths = truthfile['truth'].tolist()
        truthdict = dict(zip(items, truths))
        total_sum = 0
        for item in truthdict.keys():
            if truthdict[item] == 1:
                total_sum += 1
        correct_sum = 0
        for item in answers.keys():
            if item in truthdict.keys():
                if answers[item] == 1:
                    if answers[item] == truthdict[item]:
                        correct_sum += 1
        return correct_sum / total_sum

    def get_fscore(truthfile, answers):
        precision = get_precision(truthfile, answers)
        recall = get_recall(truthfile, answers)
        return 2 * precision * recall / (precision + recall)

    fscore = get_fscore(truthfile, answers)
    return fscore
