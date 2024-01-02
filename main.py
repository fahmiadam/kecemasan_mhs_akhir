import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

dataset_df = pd.read_csv('dataset_fix.csv')
X = dataset_df.iloc[:, :22] 
y = dataset_df['kelas']

class NWKNN:
    """
    Neighbor-weighted K-Nearest Neighbors
    """
    def __init__(self, n_neighbors=20, exp=4):
        if n_neighbors < 1:
            raise Exception('n_neighbors must be greater than 0')
        if exp <= 1:
            raise Exception('exp must be greater than 1')
            
        self.n_neighbors = n_neighbors
        self.exp = exp
        self.X_train = None
        self.y_train = None
        self.weights = None

    def __calculate_weight(self):
        counts = dict(pd.value_counts(self.y_train))
        min_count = counts[min(counts)]
        
        self.weights = dict()
        for kelas in counts:
            divide_result = counts[kelas] / min_count
            power = 1.0 / self.exp
            lower_result = pow(divide_result, power)
            self.weights[kelas] = 1.0 / lower_result
#         print(counts)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.__calculate_weight()
#         print(X_train)
#         print(y_train)
        
    
    def __decision_func(self, distances):
        distances_df = pd.DataFrame(distances, columns=['distance', 'class'])
        distances_df = distances_df.sort_values(by=['distance'])
        
        neighbors = distances_df.head(self.n_neighbors)

        # ini kalo mau pake weight
        prediction = self.__calculate_score(neighbors)

        # ini kalo ga pake weight, tinggal uncomment aja
        # prediction = neighbors['class'].value_counts().idxmax()
        return prediction


    # tinggal implementasi ini aja dam
    # return kelas prediksi berdasarkan neighbors nya
    def __calculate_score(self, neighbors):
#         print("neighbor", neighbors)
        result = []
        score_result = []

        keys = ['Tinggi', 'Rendah', 'Sedang']
        scores = []
        for index, item in neighbors.iterrows():
            temp_list = []
            temp = {
            }
            if item['distance'] != 0:
                temp[item['class']] = item['distance']
            temp_list.append(temp)
            scores.append(temp)
        
        temp = {
            "Tinggi":0,
            "Rendah":0,
            "Sedang":0
        }
        for key in keys:
            temp_result = 0
            for x in scores:
                if key in x:
                    temp_result = temp_result + x[key]
            temp[key] = temp_result
        
        score_result = temp

        result = ''
        for key in score_result:
            if score_result[key] == max(score_result.values()):
                result = key

        return result
#         print(self.weight)
        # nilai weight tinggal panggil self.weights (bentuknya udah dictionary)
        # check value : print()
        # return kelas hasil nya
#         pass


    def predict(self, X_test):
        y_pred = []
        for i, test_row in X_test.iterrows():
            distances = []
            for j, train_row in self.X_train.iterrows():
                distance = np.linalg.norm(test_row - train_row)
                kelas = y.iloc[j]
                distances.append((distance, kelas))

            prediction = self.__decision_func(distances)
            y_pred.append([prediction])
            
        return pd.Series((v[0] for v in y_pred))

def app():
    global test_df
    st.title('Klasifikasi Kecemasan')

    options = {
        'Sangat tidak sesuai': 1,
        'Tidak sesuai': 2,
        'Sesuai': 3,
        'Sangat sesuai': 4
    }

    p1 = st.selectbox(
        'P1. Saya merasa ingin buang air kecil setiap memikirkan pekerjaan setelah lulus.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p1_value = options[p1]

    p2 = st.selectbox(
        'P2. Telapak tangan saya berkeringat setiap saya memikirkan lapangan pekerjaan yang semakin sempit.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p2_value = options[p2]

    p3 = st.selectbox(
        'P3. Jantung saya berdebar-debar bila memikirkan pekerjaan setelah lulus kuliah.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p3_value = options[p3]

    p4 = st.selectbox(
        'P4. Saya kesal jika ditanya mengenai pekerjaan apa yang saya inginkan setelah lulus.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p4_value = options[p4]

    p5 = st.selectbox(
        'P5. Saya merasa pusing saat memikirkan rencana kerja setelah lulus.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p5_value = options[p5]

    p6 = st.selectbox(
        'P6. Tubuh saya gemetar memikirkan tes seleksi kerja yang akan saya hadapi.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p6_value = options[p6]

    p7 = st.selectbox(
        'P7. Tanpa sadar saya menggerakkan anggota tubuh (seperti tangan/kaki) berulang-ulang saat memikirkan pekerjaan.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p7_value = options[p7]

    p8 = st.selectbox(
        'P8. Saya mengalihkan pembicaraan apabila orang lain bertanya mengenai minat pekerjaan saya.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p8_value = options[p8]

    p9 = st.selectbox(
        'P9. Saya memilih berlama-lama kuliah karena belum siap untuk mencari pekerjaan.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p9_value = options[p9]

    p10 = st.selectbox(
        'P10. Saat hendak tidur, saya tidak mudah terlelap akibat pikiran mengenai rencana kerja.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p10_value = options[p10]

    p11 = st.selectbox(
        'P11. Saya bermimpi buruk mengenai pekerjaan.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p11_value = options[p11]

    p12 = st.selectbox(
        'P12. Saya membutuhkan orang lain untuk membantu saya mendapatkan pekerjan.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p12_value = options[p12]

    p13 = st.selectbox(
        'P13. Saya takut saya tidak mendapatkan pekerjaan yang saya inginkan.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p13_value = options[p13]

    p14 = st.selectbox(
        'P14. Saya takut kelak tidak menjadi orang yang sukses.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p14_value = options[p14]

    p15 = st.selectbox(
        'P15. Saya takut tidak lolos seleksi kerja.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p15_value = options[p15]

    p16 = st.selectbox(
        'P16. Saya khawatir tidak mendapat pekerjaan setelah lulus.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p16_value = options[p16]

    p17 = st.selectbox(
        'P17. Saya khawatir pekerjaan saya kelak tidak sesuai harapan.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p17_value = options[p17]

    p18 = st.selectbox(
        'P18. Saya gelisah mengetahui lulusan mahasiswa yang belum diterima kerja.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p18_value = options[p18]

    p19 = st.selectbox(
        'P19. Saya mudah kehilangan fokus akibat memikirkan pekerjaan saya kelak.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p19_value = options[p19]

    p20 = st.selectbox(
        'P20. Saya kehilangan konsentrasi bila teringa saya harus bekerja setelah lulus.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p20_value = options[p20]

    p21 = st.selectbox(
        'P21. Saya tidak yakin saya mampu mendapat pekerjaan sesuai harapan saya.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p21_value = options[p21]

    p22 = st.selectbox(
        'P22. Saya merasa tidak mampu bersaing dengan pencari kerja lainnya.',
        ('Sangat tidak sesuai', 'Tidak sesuai', 'Sesuai', 'Sangat sesuai')
    )
    p22_value = options[p22]


    test = [
        p1_value,
        p2_value,
        p3_value,
        p4_value,
        p5_value,
        p6_value,
        p7_value,
        p8_value,
        p9_value,
        p10_value,
        p11_value,
        p12_value,
        p13_value,
        p14_value,
        p15_value,
        p16_value,
        p17_value,
        p18_value,
        p19_value,
        p20_value,
        p21_value,
        p22_value
        #1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
    # print(test)
    # st.write(len(test))
    

    columns = []
    for i in range(1, 23):
        columns.append(f'p{i}')

    if st.button('Classify'):
        s = pd.DataFrame.from_dict({
            'p1': [p1_value], 'p2' :[p2_value],'p3': [p3_value], 'p4' :[p4_value],'p5': [p5_value], 
            'p6' :[p6_value],'p7': [p7_value], 'p8' :[p8_value],'p9': [p9_value], 'p10' :[p10_value],
            'p11': [p11_value], 'p12' :[p12_value],'p13': [p13_value], 'p14' :[p14_value],'p15': [p15_value], 
            'p16' :[p16_value],'p17': [p17_value], 'p18' :[p18_value],'p19': [p19_value], 'p20' :[p20_value],
            'p21' :[p21_value],'p22' :[p22_value]
              })
        
        st.dataframe(s)
        
        newClf = open("test.obj",'rb')
        object_file = pd.read_pickle(newClf)
        print(type(object_file))

        # st.dataframe(type(object_file))
        result = object_file.predict(s)
        print ("result", result)
        st.write(result)


if __name__ == '__main__':
    app()