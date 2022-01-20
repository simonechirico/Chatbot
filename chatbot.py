import json

#apro il file json
with open('model.json', encoding='utf-8') as f:
    corpus = json.load(f)

    print(corpus)

# PREPROCESSING DEI DATI
import spacy

nlp = spacy.load("it_core_news_lg")

dictionary = set({}) #Creo un set che conterrà il vocabolario del corpo di testo
intents = [] #Salvo tutti i nomi degli intents 

docs = [] #Conterrà i samples


for intent in corpus["intents"]: #Itero sugli intent del corpo di testo, cioè la lista degli intent
  
  for sample in intent["samples"]: #Itero su tutti i samples (che contengono a loro volta una lista)
    
    sample = sample.lower() 
    tokens = nlp(sample) 
    doc = "" #Creo un documento vuoto
    
    for token in tokens: #Itero sui token
      if not token.is_punct and not token.is_stop:
        doc+=" "+token.lemma_ # aggiungo uno spazio per separare le parole all'intermo della stringa
        dictionary.add(token.lemma_) 
        
    if(len(doc)>0): 
      docs.append(doc.rstrip()) 
      intents.append(intent["name"])
  
print("Lunghezza del dizionario: %d" % len(dictionary))
print(docs) # Documenti lemmatizzati
print(intents) # Intents

# BAG OF WORDS
# codifico il testo in numeri utilizzando una rappresentazione bag of words, uso la classe CountVectorizer di sklearn
from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer()
X = bow.fit_transform(docs)
#print(X.shape) #Dimensione array

# Gli intents sono i target della nostra rete neurale, al momento ogni intent è rappresentato da una stringa (l'identificativo), 
# uso la classe LabelEncoder per codificarli in numeri.
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(intents)
#print(y[:5]) # Stampa i primi 5 intent (sono ordinati ma dopo li mescoliamo)

# Eseguo il one hot encoding per creare le variabili di comodo per il target
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
y = ohe.fit_transform(y.reshape(-1, 1)) # Uso reshape per trasformare y da unidimensionale a bidimensionale
#print(y.shape)

y[0].toarray() # Conterrà 16 valori di cui un 1

# Mescolo il dataset usando la funzione shuffle di sklearn
from sklearn.utils import shuffle

X, y = shuffle(X, y, random_state=0)

# Creazione della rete
from keras.models import Sequential
from keras.layers import Dense

model = Sequential() #Creaiamo una RETE NEURALE vuota senza strati
model.add(Dense(12, activation="relu", input_dim=X.shape[1])) #Aggiungiamo uno strato Dense 
# (cioè ogni nodo dello strato precedente è collegato a ogni nodo dello strato successivo).
# Per gli strati nascosti la funzione di attivazione sarà la "relu" e per il primo strato 
# dobbiamo passare il numero di nodi dello strato di input.

model.add(Dense(8, activation="relu")) #Aggiungiamo un ulteriore strato nascosto con 8 nodi 
# e sempre relu come funzione di attivazione
model.add(Dense(y.shape[1], activation="softmax")) #aggiungiamo lo strato di output che deve essere uguale al numero di intent, 
# cioè il numero di colonne dell'array con i target.

#Fatto ciò l'architettura della rete neurale è pronta.

# Compiliamo il modello prima di avviare la fase di addestramento, usando come funzione di costo la categorical crossentropy 
# perchè si tratta di un problema di classificazione multiclasse e come algoritmo di ottimizzazione adam,
# che è un algoritmo di ottimizzazione che utilizza un editing rate adattivo insieme al momentum, 
# aggiungiamo anche l'accuracy come metrica.

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Avviamo l'addestramento con il metodo .fit(), impostando il numero di epoche desiderate

model.fit(X.toarray(), y.toarray(), epochs=500)

#model.fit(X, y, validation_data = (X, y), epochs = 20, batch_size = 64, shuffle = True)

# TESTIAMO IL CHATBOT
# Adesso che la nostra rete è in grado di riconsocere l'intent di una richiesta, usiamola per creare il nostro chatbot. 
# Definiamo una prima funzione che prende in ingresso la richiesta dell'utente e la processa esattamente come abbiamo 
# processato i dati dell'addestramento.

def preprocess(sentence):
  
  tokens = nlp(sentence.lower()) #Tokenizziamo e convertiamo in minuscolo
  doc = "" #Stringa vuota
  
  for token in tokens:
    if not token.is_punct and not token.is_stop:
      doc+=" "+token.lemma_

  x = bow.transform([doc.strip()]) #Usiamo solo tranform per utilizzare lo stesso modello di prima
  
  return x

from random import choice

def get_response(intent_name):
  
  for intent in corpus["intents"]: #Itera su tutti gli intent
    if intent["name"]==intent_name: #Appena trova l'intent corretto utilizza choise per estrarre dalla lista una risposta casuale tra quelle disponibili
      return choice(intent["responses"])

# Testiamo
text = "Ciao" #Input da utente
x = preprocess(text)
#print(x.shape)
import numpy as np
#utilizziamo 
y_proba = model.predict([x])[0] #il risultato sarà un array, prendiamo soltanto il primo
 #Conterrà la probabilità di appartenenza a ogniuno dei 16 target (classi o intent)

#Seleziono l'intent con la probabilità maggiore
y_proba.argmax()
print(y_proba)

#model.predict_classes([x]) # DEPRECATA

#Estraimo l'intent con il label encoded. Ritrasformiamo il numero in stringa
#y_intent = le.inverse_transform(model.predict_classes([x]))


'''
#Vediamo la risposta del nostro chatbot
get_response(y_intent)

#Creiamo il core del chatbot mettendo insieme tutto e creando una funzione che prende in input la richiesta, 
# la preprocessa, predice l'intent e ritorna la risposta.
def chatbot(sentence):
  
  x = preprocess(sentence)
  y_proba = model.predict(x)[0]
  if(y_proba.max()>.7):
    y = y_proba.argmax()
    intent = le.inverse_transform([y])
    return get_response(intent)
  else:
    return "Temo proprio di non aver capito"

#Proviamo a chattare con il nostro chatbot (per chiudere la conversazione scriviamo 'arrivederci')

sentence = ""

print("Ciao sono il tuo chatbot come posso aiutarti?")

while(sentence.lower()!="arrivederci"):
  sentence = input("Tu: ")
  response = chatbot(sentence)
  print("Chatbot: "+response)
  '''
