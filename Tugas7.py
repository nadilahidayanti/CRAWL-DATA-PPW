# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:19:33 2019

@author: NadilaHidayanti
"""
from math import log10
from requests import get
from bs4 import BeautifulSoup
import sqlite3
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")


judul=[]
isi=[]
komentar=[]


conn = sqlite3.connect('review_lensa_buku.db')
conn.execute('''CREATE TABLE if not exists BUKU
         (JUDUL VARCHAR NOT NULL,
         ISI VARCHAR NOT NULL,
         KOMENTAR VARCHAR NOT NULL);''')


src = "http://www.lensabuku.com/page/"
for cinta in range(30):
    print(cinta)
    page=get(src+"%s"%cinta)
    soup=BeautifulSoup(page.content,'html.parser')
    link = soup.findAll(class_='more-link')
    komen = soup.findAll(class_= 'entry-comments-link')
    
    for l in range(len(link)):
        page = get(link[l]['href'])
        soup = BeautifulSoup(page.content, 'html.parser')
    
        konten = soup.find(class_='entry-content')
        judul = soup.find(class_='entry-title').getText()
        komentar = komen[l].getText()
        tulisan = konten.findAll('p')
        isi = ''
        for i in tulisan:
            isi += i.getText()
        conn.execute('INSERT INTO BUKU(JUDUL,ISI,KOMENTAR) values (?, ?, ?)', (judul, isi, komentar))


conn.commit()
cursor = conn.execute("SELECT* from BUKU")
isi = []
for row in cursor:
    isi.append(row[1])
    #print(row)
    
#vsm
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover ()

factory = StemmerFactory ()
stemmer = factory.create_stemmer ()

tmp = ''
for i in isi:
    tmp = tmp + ' ' +i

hasil = []
for i in tmp.split():
    try :
        if i.isalpha() and (not i in hasil) and len(i)>1:
            # Menghilangkan Kata tidak penting
            stop = stopword.remove(i)
            if stop != "":
                stem = stemmer.stem(stop)
                hasil.append(stem)
    except:
        continue
katadasar=np.array(hasil)
print("vsm")

#KBBI
koneksi = sqlite3.connect('KBI.db')
cur_kbi = koneksi.execute("SELECT* from KATA")
    
def LinearSearch (kbi,kata):
    found=False
    posisi=0
    while posisi < len (kata) and not found :
        if kata[posisi]==kbi:
            found=True
        posisi=posisi+1
    return found
berhasil=[]
for kata in cur_kbi :
    ketemu=LinearSearch(kata[0],katadasar)
    if ketemu :
        kata = kata[0]
        berhasil.append(kata)
#print(berhasil)
katadasar = np.array(berhasil)

matrix = []
for row in isi :
    tamp_isi=[]
    for a in katadasar:
        tamp_isi.append(row.lower().count(a))
    matrix.append(tamp_isi)
print("matriks")
#print(katadasar)
#for m in matrix:
 #   print(m)

with open ('data_matrix.csv', newline='', mode='w')as employee_file :
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(katadasar)
    for i in matrix :
        employee_writer.writerow(i)

#tf-idf
df = list()
for d in range (len(matrix[0])):
    total = 0
    for i in range(len(matrix)):
        if matrix[i][d] !=0:
            total += 1
    df.append(total)

idf = list()
for i in df:
    tmp = 1 + log10(len(matrix)/(1+i))
    idf.append(tmp)

tf = matrix
tfidf = []
for baris in range(len(matrix)):
    tampungBaris = []
    for kolom in range(len(matrix[0])):
        tmp = tf[baris][kolom] * idf[kolom]
        tampungBaris.append(tmp)
    tfidf.append(tampungBaris)
tfidf = np.array(tfidf)
print("tfidf")

with open('tf-idf.csv', newline='', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(katadasar)
    for i in tfidf:
        employee_writer.writerow(i)

def pearsonCalculate(data, u,v):
    "i, j is an index"
    atas=0; bawah_kiri=0; bawah_kanan = 0
    for k in range(len(data)):
        atas += (data[k,u] - meanFitur[u]) * (data[k,v] - meanFitur[v])
        bawah_kiri += (data[k,u] - meanFitur[u])**2
        bawah_kanan += (data[k,v] - meanFitur[v])**2
    bawah_kiri = bawah_kiri ** 0.5
    bawah_kanan = bawah_kanan ** 0.5
    return atas/(bawah_kiri * bawah_kanan)
def meanF(data):
    meanFitur=[]
    for i in range(len(data[0])):
        meanFitur.append(sum(data[:,i])/len(data))
    return np.array(meanFitur)
def seleksiFiturPearson(katadasar, data, threshold):
    global meanFitur
    meanFitur = meanF(data)
    u=0
    while u < len(data[0]):
        dataBaru=data[:, :u+1]
        meanBaru=meanFitur[:u+1]
        katadasarBaru=katadasar[:u+1]
        v = u
        while v < len(data[0]):
            if u != v:
                value = pearsonCalculate(data, u,v)
                if value < threshold:
                    dataBaru = np.hstack((dataBaru, data[:, v].reshape(data.shape[0],1)))
                    meanBaru = np.hstack((meanBaru, meanFitur[v]))
                    katadasarBaru = np.hstack((katadasarBaru, katadasar[v]))
                    
            v+=1
        data = dataBaru
        meanFitur=meanBaru
        katadasar = katadasarBaru
        if u%50 == 0 : print("proses : ", u, data.shape)
        u+=1
    return katadasar, data

katadasarBaru, fiturBaru = seleksiFiturPearson(katadasar, tfidf, 0.8)

for i in range(2, len(fiturBaru)-1):
    
    kmeans = KMeans(n_clusters=i, random_state=0).fit(fiturBaru);
    
    classnya = kmeans.labels_
    s_avg = silhouette_score(fiturBaru, classnya, random_state=0)
    
    print("Silhouette untuk", i, "cluster adalah",s_avg)
    print(kmeans.labels_)
print("finish")
with open('Anggota_cluster.csv', newline='', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in classnya.reshape(-1,1):
        employee_writer.writerow(i)

with open('Seleksi_Fitur.csv', newline='', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow([katadasarBaru.tolist()])
    for i in fiturBaru:
        employee_writer.writerow(i)

