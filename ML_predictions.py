import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

with open('data.pkl', 'rb') as pickle_file:
    data = pickle.load(pickle_file)
with open('prices.pkl', 'rb') as prices_file:
    prices = pickle.load(prices_file)

rows, columns = data.shape
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 0:columns-1], data['Actions'], test_size=0.33, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Classifier prediction accuracy: ", clf.score(X_test, y_test))

# Adjust actions so that cannot buy twice in a row etc. and first actions is a buy
actions = list(predictions)
first_buy = actions.index(-1)
for i in range(first_buy):
    actions[i] = 0

last_action = 0
for i in range(len(actions)):
    # Correct any double buys / double sells
    if last_action == -1 and actions[i] == -1:
        actions[i] = 0
    if last_action == 1 and actions[i] == 1:
        actions[i] = 0
    # Update last_action
    if actions[i] == 1:
        last_action = 1
    elif actions[i] == -1:
        last_action = -1

# Plot
x_axis = np.linspace(0, len(predictions), len(predictions))
fig, ax = plt.subplots()
markers = ['cyan' if i==0 else 'green' if i==-1 else 'red' for i in actions]
plt.scatter(x_axis, X_test['Open'], color=markers)
plt.plot(x_axis, X_test['Open'])
plt.show()

money = 1
transactions = 0
last_action_index = first_buy
for i in range(len(actions)):
    # if transactions == 0 and (actions[i] == 1 or actions[i] == -1):
    #     pass
    # else:
    if actions[i] == 1:
        delta = float(prices.iloc[i, 0]) / float(prices.iloc[last_action_index, 0])
        money = money * delta * 0.999
    elif actions[i] == -1:
        last_action_index = i
        money = money * 0.999

print("Your return is:", (money - 1) * 100, "%")