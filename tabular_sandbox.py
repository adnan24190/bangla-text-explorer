from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
X,y = load_breast_cancer(return_X_y=True)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
dummy = DummyClassifier(strategy = "most_frequent")
log_reg = LogisticRegression()
dummy.fit(X_train, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log_reg.fit(X_train_scaled, y_train)
print("Dummy Score:", dummy.score(X_test, y_test))
print("LogReg Score:", log_reg.score(X_test_scaled, y_test))
print("Predictions:", log_reg.predict(X_test_scaled[:10]))
print("Actuals:    ", y_test[:10])