import sys, os, math
sys.path.append(os.getcwd())
from perceptron import AveragedPerceptronClassifier

errs = 0
clf = AveragedPerceptronClassifier(train_dir = 'haiti/train')
clf.train()
nostop_clf = AveragedPerceptronClassifier(train_dir = 'haiti/train', REMOVE_STOPWORDS=True)
nostop_clf.train()
print("Checking vocabulary sizes...")
verr = False
try:
    assert len(clf.vocabulary) == 3246
except AssertionError:
    print(f'\terror, expected vocab size of 3246 including stop words but got {len(clf.vocabulary)}')
    errs += 1
    verr = True
try:
    assert len(nostop_clf.vocabulary) == 2909
except AssertionError:
    print(f'\terror, expected vocab size of 2909 without stop words but got {len(nostop_clf.vocabulary)}')
    errs += 1
    verr = True
if not verr:
    print('\tlooks good!')

print("Checking weights...")
werr = False
try:
    val = clf.weights['hope']
    assert math.fabs(1.0004 + val) < 0.05
except:
    errs += 1
    werr = True

try:
    val = nostop_clf.weights['give']
    assert math.fabs(0.3359 - val) < 0.05
except:
    errs += 1
    werr = True

if not werr:
    print('\tlooks good!')
else:
    print('\terror with weights, values incorrect')

print('Checking document scoring...')
try:
    doc = 'i really need help'
    score_val = 0
    for word in doc.split():
        if not word in clf.vocabulary: continue
        score_val += clf.weights[word]
    assert math.fabs(8.0644 + score_val) < 0.5
    print('\tlooks good!')
except AssertionError:
    errs += 1
    print(f'\terror with document scoring')

print('Checking evaluation on the training set...')
precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/train', target = 'relevant')
try:
    assert math.fabs(precision - 0.970) < 0.01
    assert math.fabs(recall - 0.989) < 0.01
    assert math.fabs(f1_score - 0.980) < 0.01
    print('\tlooks good!')
except AssertionError:
    errs += 1
    print('\terror with evaluation on the training set!')
    print('\texpected P~0.970, R~0.989, F1~0.980')
    print(f'\tgot P~{precision}, R~{recall}, F1~{f1_score}')

print('Checking evaluation on the test set...')
precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/dev', target = 'relevant')

try:
    assert math.fabs(precision - 0.837) < 0.01
    assert math.fabs(recall - 0.957) < 0.01
    assert math.fabs(f1_score - 0.893) < 0.01
    print('\tlooks good!')
except AssertionError:
    errs += 1
    print('\terror with evaluation on the test set!')
    print('\texpected P~0.837, R~0.957, F1~0.893')
    print(f'\tgot P~{precision}, R~{recall}, F1~{f1_score}')

if errs == 0:
    print('All tests passed!\n')
else:
    print("... looks like there's still a few things to fix.")
