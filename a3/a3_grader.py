import sys, os, math
sys.path.append(os.getcwd())
from naive_bayes import NaiveBayesClassifier

print("Training both models...")
clf = NaiveBayesClassifier(train_dir = 'haiti/train')
clf.train()
nostop_clf = NaiveBayesClassifier(train_dir = 'haiti/train', REMOVE_STOPWORDS=True)
nostop_clf.train()
print("Checking vocabulary sizes...")
errors = 0
try:
    assert len(clf.vocabulary) == 3246
except AssertionError:
    print(f'\terror, expected vocab size of 3246 including stop words but got {len(clf.vocabulary)}')
    errors += 1
try:
    assert len(nostop_clf.vocabulary) == 2909
except AssertionError:
    print(f'\terror, expected vocab size of 2909 without stop words but got {len(nostop_clf.vocabulary)}')
    errors += 1
if errors == 0:
    print('\tlooks good!')

print("Checking log priors...")

try:
    assert math.fabs(0.410118 + clf.logprior['relevant']) < 0.01
    print('\tlooks good!')
except:
    if 'relevant' in clf.logprior:
        print(f"\terror, expected log prior of about -0.410118 for 'relevant' class but got {clf.logprior['relevant']}")
    else:
        print("\terror, logprior is missing the key for the 'relevant' class")
    errors += 1

print("Checking log likelihoods...")
ll_err = False
try:
    val = clf.loglikelihood['hope', 'relevant']
    assert math.fabs(8.15966 + val) < 0.05
except:
    errors += 1
    ll_err = True
try:
    val = nostop_clf.loglikelihood['give', 'irrelevant']
    assert math.fabs(5.50256 + val) < 0.05
except:
    errors += 1
    ll_err = True

if not ll_err:
    print('\tlooks good!')
else:
    print('\terror with log likelihoods, either keys missing or values incorrect')

print('Checking document scoring...')
try:
    val = clf.score('i really need help','relevant')
    assert math.fabs(20.5276 + val) < 0.5
    print('\tlooks good!')
except AssertionError:
    errors += 1
    print(f'\terror with document scoring, expected a score of around -20.5276 and got {val}')
    
print('Checking evaluation on the training set...')
precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/train', target = 'relevant')
try:
    assert math.fabs(precision - 0.9612289) < 0.01 or math.fabs(precision - 0.944151) < 0.01
    assert math.fabs(recall - 0.97550111) < 0.01 or math.fabs(recall - 0.964477) < 0.01
    assert math.fabs(f1_score - 0.9683124) < 0.01 or math.fabs(f1_score - 0.954206) < 0.01
    print('\tlooks good!')
except AssertionError:
    errors += 1
    print('\terror with evaluation on the training set!')
    print('\texpected P~0.961, R~0.976, F1~0.968')
    print(f'\tgot P~{precision}, R~{recall}, F1~{f1_score}')

print('Checking evaluation on the test set...')
precision, recall, f1_score = clf.evaluate(test_dir = '/Users/caroclaeson/PycharmProjects/compling/a3/haiti_test', target = 'relevant')
try:
    assert math.fabs(precision - 0.8466899) < 0.01 or math.fabs(precision - 0.795349) < 0.01
    assert math.fabs(recall - 0.9455252918) < 0.01 or math.fabs(recall - 0.9243243) < 0.01
    assert math.fabs(f1_score - 0.893382352941176) < 0.01 or math.fabs(f1_score - 0.8549999) < 0.01
    print('\tlooks good!')
except AssertionError:
    errors += 1
    print('\terror with evaluation on the test set!')
    print('\texpected P~0.847, R~0.946, F1~0.893')
    print(f'\tgot P~{precision}, R~{recall}, F1~{f1_score}')

if errors == 0:
    print('All tests passed!\n')
else:
    print("... looks like there's still a few things to fix.")
