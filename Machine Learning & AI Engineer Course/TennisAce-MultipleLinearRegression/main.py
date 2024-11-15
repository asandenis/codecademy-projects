import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/tennis_stats.csv')

print('First entries of the dataframe:\n', df.head())

figure, axis = plt.subplots(2, 2)

y = df[['Winnings']]
x = df[['DoubleFaults']]
x2 = df[['Aces']]

# Linear Regression - DoubleFaults feature.

axis[0, 0].plot(x, y, alpha=0.5)

axis[0, 0].set_xlabel('DoubleFaults')
axis[0, 0].set_ylabel('Winnings')

axis[0, 0].set_title('Linear Regression - DoubleFaults feature')

x_lr1_train, x_lr1_test, y_lr1_train, y_lr1_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=6)

lr1 = LinearRegression()

lr1.fit(x_lr1_train, y_lr1_train)

y_pred = lr1.predict(x_lr1_test)

print('\nLinear Regression - DoubleFaults feature - train set R-squared:')
print("%.2f" % (100*lr1.score(x_lr1_train, y_lr1_train)), '%')
print('Linear Regression - DoubleFaults feature - test set R-squared:')
print("%.2f" % (100*lr1.score(x_lr1_test, y_lr1_test)), '%')

# Linear Regression - Aces feature.

axis[0, 1].plot(x2, y, alpha=0.5)

axis[0, 1].set_xlabel('Aces')
axis[0, 1].set_ylabel('Winnings')

axis[0, 1].set_title('Linear Regression - Aces feature')

x_lr2_train, x_lr2_test, y_lr2_train, y_lr2_test = train_test_split(x2, y, train_size=0.8, test_size=0.2, random_state=6)

lr2 = LinearRegression()

lr2.fit(x_lr2_train, y_lr2_train)

y2_pred = lr2.predict(x_lr2_test)

print('\nLinear Regression - Aces feature - train set R-squared:')
print("%.2f" % (100*lr2.score(x_lr2_train, y_lr2_train)), '%')
print('Linear Regression - Aces feature - test set R-squared:')
print("%.2f" % (100*lr2.score(x_lr2_test, y_lr2_test)), '%')

# Multiple Linear Regression - Aces & DoubleFaults features.

mlr1 = LinearRegression()

x_mlr1 = df[['Aces', 'DoubleFaults']]

axis[1, 0].plot(x_mlr1, y, alpha=0.5)

axis[1, 0].set_xlabel('Aces & DoubleFaults')
axis[1, 0].set_ylabel('Winnings')

axis[1, 0].set_title('Multiple Linear Regression - Aces & DoubleFaults features')

x_mlr1_train, x_mlr1_test, y_mlr1_train, y_mlr1_test = train_test_split(x_mlr1, y, train_size=0.8, test_size=0.2, random_state=6)

mlr1.fit(x_mlr1_train, y_mlr1_train)

y_mlr1_pred = mlr1.predict(x_mlr1_test)

print('\nMultiple Linear Regression - Aces & DoubleFaults features - train set R-squared:')
print("%.2f" % (100*mlr1.score(x_mlr1_train, y_mlr1_train)), '%')
print('Multiple Linear Regression - Aces & DoubleFaults features - test set R-squared:')
print("%.2f" % (100*mlr1.score(x_mlr1_train, y_mlr1_train)), '%')

# Multiple Linear Regression - Aces, DoubleFaults, FirstServe & BreakPointsOpportunities features.

x_mlr2 = df[['Aces', 'DoubleFaults', 'FirstServe', 'BreakPointsOpportunities']]

axis[1, 1].plot(x_mlr2, y, alpha=0.5)

axis[1, 1].set_xlabel('Aces, DoubleFaults, FirstServe & BreakPointsOpportunities')
axis[1, 1].set_ylabel('Winnings')

axis[1, 1].set_title('Multiple Linear Regression - Aces, DoubleFaults, FirstServe & BreakPointsOpportunities features')

mlr2 = LinearRegression()

x_mlr2_train, x_mlr2_test, y_mlr2_train, y_mlr2_test = train_test_split(x_mlr2, y, train_size=0.8, test_size=0.2, random_state=6)

mlr2.fit(x_mlr2_train, y_mlr2_train)

y_mlr2_pred = mlr2.predict(x_mlr2_test)

print('\nMultiple Linear Regression - Aces, DoubleFaults, FirstServe & BreakPointsOpportunities features - train set R-squared:')
print("%.2f" % (100*mlr2.score(x_mlr2_train, y_mlr2_train)), '%')
print('Multiple Linear Regression - Aces, DoubleFaults, FirstServe & BreakPointsOpportunities features - test set R-squared:')
print("%.2f" % (mlr2.score(x_mlr2_test, y_mlr2_test)), '%')

plt.show()