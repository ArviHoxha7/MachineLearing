import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('project2_dataset.csv')

# 1. Εγγραφές συνόλου δεδομένων
num_records = df.shape[0]
print(f'Αριθμός εγγραφών: {num_records}')

# 2. Ποσοστό όπου οι χρήστες αγόρασαν
purchase_rate = df['Revenue'].mean() * 100
print(f'Ποσοστό χρηστών που αγόρασαν: {purchase_rate:.2f}%')

# 3. Ευστοχία ενός μοντέλου το οποίο προβλέπει πάντα ότι ο χρήστης δε θα αγοράσει
accuracy_no_purchase_model = (df['Revenue'] == 0).mean() * 100
print(f'Ακρίβεια μοντέλου που προβλέπει πάντα ότι δε θα γίνει αγορά: {accuracy_no_purchase_model:.2f}%')

# Προαιρετικά διαγράμματα
sns.countplot(x='Revenue', data=df)
plt.title('Κατανομή της μεταβλητής στόχου (Revenue)')
plt.show()

sns.histplot(df['Region'], kde=True)
plt.title('Κατανομή της μεταβλητής Region')
plt.show()
