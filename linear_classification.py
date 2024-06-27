from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def prepare_data(df, train_size=0.7, shuffle=True, random_state=42):
    # Αφαίρεση των χαρακτηριστικών
    df = df.drop(columns=['Month', 'Browser', 'OperatingSystems'])

    # Μετατροπή boolean τιμών σε αριθμητικές
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)

    # One-hot encoding στις μεταβλητές
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])

    # Χωρισμός της μεταβλητής στόχου από τις υπόλοιπες
    X = df.drop(columns=['Revenue'])
    y = df['Revenue']

    # Χωρισμός του συνόλου δεδομένων σε εκπαίδευσης και δοκιμής
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)

    return X_train, X_test, y_train, y_test


df = pd.read_csv('project2_dataset.csv')
# Κλήση της συνάρτησης prepare_data με τις κατάλληλες παραμέτρους
X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.7, random_state=42)

# Εφαρμογή γραμμικής κανονικοποίησης
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Τα δεδομένα προετοιμάστηκαν και κανονικοποιήθηκαν.")

# Δημιουργία του μοντέλου LogisticRegression
model = LogisticRegression(max_iter=1000)

# Εκπαίδευση του μοντέλου
model.fit(X_train_scaled, y_train)

# Προβλέψεις στα σύνολα εκπαίδευσης και δοκιμής
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("Το μοντέλο εκπαιδεύτηκε και έγιναν οι προβλέψεις.")

# Υπολογισμός και εκτύπωση της ακρίβειας του μοντέλου
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Ακρίβεια στο σύνολο εκπαίδευσης: {train_accuracy:.2f}')
print(f'Ακρίβεια στο σύνολο δοκιμής: {test_accuracy:.2f}')

# Υπολογισμός και εκτύπωση του πίνακα σύγχυσης
conf_matrix = confusion_matrix(y_test, y_test_pred)
print('Πίνακας σύγχυσης:')
print(conf_matrix)

tn, fp, fn, tp = conf_matrix.ravel()
print(f'True Negatives: {tn}')
print(f'False Positives: {fp}')
print(f'False Negatives: {fn}')
print(f'True Positives: {tp}')
