train_counts = list()
y_train_dtc = list(y_train)
for i in range (62):
    train_counts.append(y_train_dtc.count(i))
for i in range (len(y_train_dtc)):
    y_train_dtc[i] = [y_train_dtc[i], i]
y_train_dtc.sort()
x_train_balanced = list()
y_train_balanced = list()
n = 1895
intervalos = list()
for i in range (62):
    inf = sum(train_counts[:i]) + 1
    inf_mas_1 = sum(train_counts[:i+1])
    sup = min(inf + n, inf_mas_1)
    intervalos.append([inf, sup])
for i in range (62):
    y_train_balanced.extend(y_train_dtc[intervalos[i][0]:intervalos[i][1]])
shuffle(y_train_balanced)
for i in range (len(y_train_balanced)):
    x_train_balanced.append(x_train_dtc[y_train_balanced[i][1]])
    y_train_balanced[i] = y_train_balanced[i][0]
clases62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

inicio = time.time()
clf_dtc2 = GaussianNB()
clf_dtc2.fit(x_train_balanced, y_train_balanced)
fin = time.time()
print(fin - inicio)

y_test_dtc = list(y_test)
test_counts = list()
for i in range (61):
    test_counts.append(y_test_dtc.count(i))
for i in range (len(y_test)):
    y_test_dtc[i] = [y_test[i], i]
y_test_dtc.sort()
x_test_balanced = list()
y_test_balanced = list()
n = 317
intervalos = list()
for i in range (62):
    inf = sum(test_counts[:i]) + 1
    sup = inf + n
    intervalos.append([inf, sup])
for i in range (62):
    y_test_balanced.extend(y_test_dtc[intervalos[i][0]:intervalos[i][1]])
shuffle(y_test_balanced)
for i in range (len(y_test_balanced)):
    x_test_balanced.append(x_test[y_test_balanced[i][1]])
    y_test_balanced[i] = y_test_balanced[i][0]

predicted_dtc = clf_dtc.predict(x_test_balanced)
expected_dtc = y_test_balanced
predicted_dtc2 = clf_dtc2.predict(x_test_balanced)
expected_dtc2 = y_test_balanced

print(metrics.classification_report(expected_dtc2, predicted_dtc2))

df = pd.DataFrame(metrics.confusion_matrix(expected_dtc2, predicted_dtc2))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(df)

df = pd.DataFrame(metrics.confusion_matrix(expected_dtc, predicted_dtc))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(df)