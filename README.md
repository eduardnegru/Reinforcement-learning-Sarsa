# Reinforcement-learning-Sarsa

**Cerinta 1**

Am implementat algoritmul Sarsa si cele 3 strategii de explorare: epsilon-greedy, Boltzmann, upper confidence bound. Algoritmul
este in fisierul sarsa_skel.py. Algoritmul a fost testat folosind python3.6.

**Cerinta2**

Am facut diverse teste pentru c = [0.9, 0.5, 0.1] si alpha = [0.1, 0.05, 0.01] atat pentru epsilon-greedy cat si pentru softmax. Testele
Au fost testate toate posibilitatile de a alege o pereche (c, alpha) din cele 2 liste. Un exemplu de script de testare
pe care l-am folosit este in fisierul test.py. Testele pe 8x8_doorkey si 16x16_doorkey dureaza foarte mult si au nevoie de
un numar foarte mare de pasi pentru a reusi sa invete.(dureaza aproximativ 3h pentru 30k pasi pentru 8x8; nu am rulat pentru 
8x8_doorkey, 16x16_doorkey pentru ca dureaza prea mult).
In urma rularii testelor am determinat ca hiper-parametrii cei mai potriviti sunt alpha=0.01, c=0.05, gamma = 0.9.

Rezultatele testelor pot fi vazute in fisierele: 6x6_empty, 8x8_empty, 16x16_empty, 6x6_key.

**Cerinta 3** 
  
Am testat o alta valoare pentru q0. Am folosit q0 = 5. Am obtinut un average return maxic mai mic decat in cazul in care 
q0 = 0. De asemenea, timpul de rulare pentru q0=5 a crescut de la aproximativ 4 minute(pentru q0 = 0) la aproximativ 15 minute
pentru 50k pasi pentru harta 8x8_empty.

Resultatele pot fi vazute in folderul q0_values.


**Cerinta 4** 

Valori hiper-parametrii: alpha=0.01, c=0.05, gamma=0.9.
Graficele sunt in directorul cerinta4.

**Bonus**

Performanta ucb este similara cu performanta e-greedy. Pentru testare am folosit constanta c=0.05.
Rezultatele obtinute sunt in directorul bonus.
