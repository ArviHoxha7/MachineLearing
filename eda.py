import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Φορτώνουμε τα δεδομένα
df = pd.read_csv('project2_dataset.csv')

# 1. Πόσες είναι οι εγγραφές του συνόλου δεδομένων;
num_records = df.shape[0]
print(f'Αριθμός εγγραφών: {num_records}')

# 2. Σε τι ποσοστό από αυτές οι χρήστες αγόρασαν τελικά;
purchase_rate = df['Revenue'].mean() * 100
print(f'Ποσοστό χρηστών που αγόρασαν: {purchase_rate:.2f}%')

# 3. Ποια είναι η ευστοχία ενός μοντέλου το οποίο προβλέπει πάντα ότι ο χρήστης δε θα αγοράσει;
accuracy_no_purchase_model = (df['Revenue'] == 0).mean() * 100
print(f'Ακρίβεια μοντέλου που προβλέπει πάντα ότι δε θα γίνει αγορά: {accuracy_no_purchase_model:.2f}%')

# Προαιρετικά διαγράμματα
# Κατανομή της μεταβλητής στόχου
sns.countplot(x='Revenue', data=df)
plt.title('Κατανομή της μεταβλητής στόχου (Revenue)')
plt.show()

# Κατανομή χαρακτηριστικών
sns.histplot(df['Region'], kde=True)
plt.title('Κατανομή της μεταβλητής Region')
plt.show()

# Σχέση χαρακτηριστικών με τη μεταβλητή στόχο
sns.boxplot(x='Revenue', y='PageValues', data=df)
plt.title('Σχέση της μεταβλητής PageValues με τη μεταβλητή στόχο')
plt.show()


# Προετοιμασία δεδομένων
def prepare_data(df, train_size=0.7, shuffle=True, random_state=42):
    # Αφαίρεση των χαρακτηριστικών Month, Browser, OperatingSystems
    df = df.drop(columns=['Month', 'Browser', 'OperatingSystems'])

    # Μετατροπή boolean τιμών σε αριθμητικές
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)

    # One-hot encoding στις μεταβλητές Region, TrafficType, VisitorType
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])

    # Χωρισμός της μεταβλητής στόχου από τις υπόλοιπες
    X = df.drop(columns=['Revenue'])
    y = df['Revenue']

    # Χωρισμός του συνόλου δεδομένων σε εκπαίδευσης και δοκιμής
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)

    return X_train, X_test, y_train, y_test

# Κλήση της συνάρτησης prepare_data για 70%-30% χωρισμό σε σύνολο εκπαίδευσης και δοκιμής και σπόρο 42
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

# Ερμηνεία του πίνακα σύγχυσης
tn, fp, fn, tp = conf_matrix.ravel()
print(f'Αληθώς Αρνητικά (TN): {tn}')
print(f'Ψευδώς Θετικά (FP): {fp}')
print(f'Ψευδώς Αρνητικά (FN): {fn}')
print(f'Αληθώς Θετικά (TP): {tp}')

# Προτάσεις για βελτιώσεις
print('Προτάσεις για βελτιώσεις:')
print('1. Δοκιμάστε άλλους αλγορίθμους ταξινόμησης όπως Random Forest ή SVM.')
print('2. Προσθέστε χαρακτηριστικά ή χρησιμοποιήστε χαρακτηριστικά που αφαιρέθηκαν.')
print('3. Χρησιμοποιήστε τεχνικές υπερσυντονισμού παραμέτρων (hyperparameter tuning).')
print('4. Δοκιμάστε τεχνικές αύξησης δεδομένων (data augmentation) ή επεξεργασίας δεδομένων.')
