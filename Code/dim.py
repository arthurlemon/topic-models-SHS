import gensim
import tqdm
import json
from gensim import corpora, utils
from gensim.models.wrappers.dtmmodel import DtmModel
from tqdm import tqdm

"""Ecriture des fonctions"""

# Récupérer les articles en français, de type article et avec année de publication
def filtre_corpus(Corpus, tokens):
    """Prend en paramètre le Corpus avec textes et métadonnées et retourne le Corpus filtré avec les articles en français et de type article
    + les tokens nettoyés pour chacun d'entre eux et l'année de publication"""
    Corpus_filtre = []
    index_clean = 0
    for index_document in tqdm(range(len(Corpus))) :
        # on ne traite que les articles de type article et en français
        if (Corpus[index_document]['metadata']['typeart'], Corpus[index_document]['metadata']['lang']) == ('article','fr') :
            Corpus_filtre.append((Corpus[index_document]['metadata']['annee'], tokens[index_clean]))
            index_clean +=1
    return Corpus_filtre


def tri_annee(Corpus_filtre):
    """On trie le corpus selon les années
        Output :
        - time_slices = liste où chaque élément contient le nombre d'articles par année
        - articles_par_annee = liste où chaque élément est un tuple du type (année, [liste des tokens des articles de cette année])"""
    time_slices = []  # chaque élément compte le nombre d'articles à chaque time step dans l'ordre croissant des années
    articles_par_annee = {}  # contient les tokens des articles pour chaque année de publication
    for element in Corpus_filtre:
        annee = element[0]
        if annee in articles_par_annee:
            articles_par_annee[annee].append(element[1])
        else:
            articles_par_annee[annee] = [element[1]]

    articles_par_année_sorted = sorted(articles_par_annee.items())  # ordonnement par année

    for annee in range(len(articles_par_année_sorted)):
        time_slices.append(len(articles_par_année_sorted[annee][1]))
    return time_slices, articles_par_année_sorted

class DTMcorpus(corpora.textcorpus.TextCorpus):

    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)


if __name__ =='__main__':

    # CODE POUR ACTUALITE ECONOMIQUE
    Corpus_AE = json.loads(open("corpus_ae.json", "r").read())
    tokens_AE= json.loads(open("tokens_ae.json", "r").read())
    Corpus_annees_AE = filtre_corpus(Corpus_AE, tokens_AE)
    time_slices_AE, articles_par_annee_AE = tri_annee(Corpus_annees_AE)
    train_texts_AE = []
    for annee in range(len(articles_par_annee_AE)):
        train_texts_AE.append(articles_par_annee_AE[annee][1])
    train_texts_AE = [element for sublist in train_texts_AE for element in sublist]

    time_slices_AE_toy = time_slices_AE # CHANGER pour code sur ordi local

    train_texts_AE_toy = []
    indice_courant = 0
    for indice in time_slices_AE: # CHANGER pour code sur ordi local
        train_texts_AE_toy.append(train_texts_AE[indice_courant:indice_courant + indice])
        indice_courant += indice

    train_texts_AE_toy = [element for sublist in train_texts_AE_toy for element in sublist]
    corpus_AE_toy = DTMcorpus(train_texts_AE_toy)
  
    print("Corpus AE pret pour l'entrainement")
    
    # CODE POUR RELATIONS INDUSTRIELLES
    Corpus_RI = json.loads(open("corpus_ri.json", "r").read())
    tokens_RI = json.loads(open("tokens_ri.json", "r").read())
    Corpus_annees_RI = filtre_corpus(Corpus_RI, tokens_RI)
    time_slices_RI, articles_par_annee_RI = tri_annee(Corpus_annees_RI)
    train_texts_RI = []
    for annee in range(len(articles_par_annee_RI)):
        train_texts_RI.append(articles_par_annee_RI[annee][1])
    train_texts_RI = [element for sublist in train_texts_RI for element in sublist]

    time_slices_RI_toy = time_slices_RI  # CHANGER pour code sur ordi local

    train_texts_RI_toy = []
    indice_courant = 0
    for indice in time_slices_RI:  # CHANGER pour code sur ordi local
        train_texts_RI_toy.append(train_texts_RI[indice_courant:indice_courant + indice])
        indice_courant += indice

    train_texts_RI_toy = [element for sublist in train_texts_RI_toy for element in sublist]
    corpus_RI_toy = DTMcorpus(train_texts_RI_toy)
  
    print("Corpus RI pret pour l'entrainement")
   
    # CODE POUR ETUDES INTERNATIONALES
    Corpus_EI = json.loads(open("corpus_ei.json", "r").read())
    tokens_EI = json.loads(open("tokens_ei.json", "r").read())
    Corpus_annees_EI = filtre_corpus(Corpus_EI, tokens_EI)
    time_slices_EI, articles_par_annee_EI = tri_annee(Corpus_annees_EI)
    train_texts_EI = []
    for annee in range(len(articles_par_annee_EI)):
        train_texts_EI.append(articles_par_annee_EI[annee][1])
    train_texts_EI = [element for sublist in train_texts_EI for element in sublist]

    time_slices_EI_toy = time_slices_EI # CHANGER pour code sur ordi local

    train_texts_EI_toy = []
    indice_courant = 0
    for indice in time_slices_EI: # CHANGER pour code sur ordi local
        train_texts_EI_toy.append(train_texts_EI[indice_courant:indice_courant + indice])
        indice_courant += indice

    train_texts_EI_toy = [element for sublist in train_texts_EI_toy for element in sublist]
    corpus_EI_toy = DTMcorpus(train_texts_EI_toy)
    
    print("Corpus EI pret pour l'entrainement")
    
    dtm_path = "/scratch/arthur33/dtm-linux64" # path pour Calcul Canada
    #dtm_path = 'dtm/dtm-win64.exe' # path pour Windows
    num_topics = 10

    print("Début de l'entrainement du modèle DTM pour AE \n")
    #model_DTM = DtmModel(dtm_path, corpus_AE_toy, time_slices_AE_toy, num_topics=num_topics, id2word=corpus_AE_toy.dictionary,initialize_lda=True)
    #model_DTM.save('dtm_ae_10')
    print("Fin de l'entrainement du modèle DTM pour AE\n")

    print("Début de l'entrainement du modèle DTM pour RI \n")
    #model_DTM = DtmModel(dtm_path, corpus_RI_toy, time_slices_RI_toy, num_topics=num_topics, id2word=corpus_RI_toy.dictionary, initialize_lda=True)
    #model_DTM.save('dtm_ri_10')
    print("Fin de l'entrainement du modèle DTM pour RI\n")

    print("Début de l'entrainement du modèle DTM pour EI \n")
    #model_DTM = DtmModel(dtm_path, corpus_EI_toy, time_slices_EI_toy, num_topics=num_topics, id2word=corpus_EI_toy.dictionary,initialize_lda=True)
    #model_DTM.save('dtm_ei_10')
    print("Fin de l'entrainement du modèle DTM pour EI\n")

    print("\n---------------------\n")

    print("Début de l'entrainement du modèle DIM pour AE \n")
    #model_DIM = DtmModel(dtm_path, corpus_AE_toy, time_slices_AE_toy, num_topics=num_topics, id2word=corpus_AE_toy.dictionary, initialize_lda=True, model='fixed')
    #model_DIM.save('dim_ae_10')
    print("Fin de l'entrainement du modèle DIM pour AE\n")

    print("Début de l'entrainement du modèle DIM pour RI \n")
    model_DIM = DtmModel(dtm_path, corpus_RI_toy, time_slices_RI_toy, num_topics=num_topics, id2word=corpus_RI_toy.dictionary,
                         initialize_lda=True, model='fixed')
    model_DIM.save('dim_ri_10')
    print("Fin de l'entrainement du modèle DIM pour RI\n")

    print("Début de l'entrainement du modèle DIM pour EI \n")
    model_DIM = DtmModel(dtm_path, corpus_EI_toy, time_slices_EI_toy, num_topics=num_topics, id2word=corpus_EI_toy.dictionary,
                         initialize_lda=True, model='fixed')
    model_DIM.save('dim_ei_10')
    print("Fin de l'entrainement du modèle DIM pour EI\n")