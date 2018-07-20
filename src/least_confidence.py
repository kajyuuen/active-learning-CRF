def calculate_least_confidence(predict_marginal):
    sequence_probability = 1
    for probabilities in predict_marginal:
        sequence_probability *= max(probabilities.values())
    return 1 - sequence_probability

def calculate_least_confidences(predict_marginals):
    least_confidences = []
    for predict_marginal in predict_marginals:
        least_confidences.append(calculate_least_confidence(predict_marginal))
    return least_confidences
