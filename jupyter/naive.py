from skmultiflow.data import SEAGenerator
from skmultiflow.bayes import NaiveBayes

# Setup a data stream
stream = SEAGenerator(random_state=1)

# Setup Naive Bayes estimator
naive_bayes = NaiveBayes()

# Setup variables to control loop and track performance
n_samples = 0
correct_cnt = 0
max_samples = 200

# Train the estimator with the samples provided by the data stream

while n_samples < max_samples and stream.has_more_samples():
    X, y = stream.next_sample()
    y_pred = naive_bayes.predict(X)

    if y[0] == y_pred[0]:
        correct_cnt += 1

    naive_bayes.partial_fit(X, y)
    n_samples += 1

# Display results
print('{} samples analyzed.'.format(n_samples))
print('Naive Bayes accuracy: {}'.format(correct_cnt / n_samples))
