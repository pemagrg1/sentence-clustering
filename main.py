import nltk
from SentenceClustering import SentenceClustering
import pandas as pd

text = """
        If I was your boyfriend, I'd never let you go. I can take you places you ain't never been before. Baby take a chance or you'll never ever know. I got money in my hands that I'd really like to blow. Swag swag swag, on you. Chillin' by the fire while we eating fondue. I don't know 'bout me but I know about you. So say hello to falsetto in three two. I'd like to be everything you want. Hey girl, let me talk to you. If I was your boyfriend, I'd never let you go. Keep you on my arm girl, you'd never be alone. I can be a gentleman, anything you want. If I was your boyfriend, I'd never let you go, I'd never let you go. Tell me what you like yeah tell me what you don't. I could be your Buzz Lightyear, fly across the globe. I don't never wanna fight yeah, you already know. I am 'ma a make you shine bright like you're laying in the snow burr. Girlfriend, girlfriend, you could be my girlfriend. You could be my girlfriend until the, world ends. Make you dance do a spin and a twirl and. Voice goin' crazy on this hook like a whirl wind swaggie. I'd like to be everything you want. Hey girl, let me talk to you. If I was your boyfriend, I'd never let you go. Keep you on my arm girl, you'd never be alone. I can be a gentleman, anything you want. If I was your boyfriend, I'd never let you go, I'd never let you go. So give me a chance, 'cause you're all I need girl. Spend a week wit' your boy I'll be calling you my girlfriend. If I was your man, I'd never leave you girl. I just want to love you, if I was your boyfriend. I'd never let you go (and treat you right). Keep you on my arm girl, you'd never be alone. I can be a gentleman, anything you want. If I was your boyfriend, I'd never let you go, I'd never let you go. Na na na na na na na na na yeah girl,. Na na na na na na na na na, if I was your boyfriend. Na na na na na na na na na hey. Na na na na na na na na na, if I was your boyfriend
        """

sents = nltk.sent_tokenize(text)
nclusters= 5
sent_clus = SentenceClustering(sents=sents, nclusters=nclusters, visualization=True)
clusters = sent_clus.kmeans_clustering()

"""
Other alog you can try:
        clusters = sent_clus.kmeans_clustering()
        clusters = sent_clus.Agglomerative_clustering()
        clusters = sent_clus.affinity_clustering()
"""

df = pd.DataFrame(columns=['sentence','label'])
for cluster in range(len(clusters.keys())):
        print ("cluster ",cluster,":")
        for i,sentence in enumerate(clusters[cluster]):
                print ("\tsentence ",i,": ",sents[sentence])
                df = df.append({'sentence': sents[sentence], 'label': cluster},
                               ignore_index=True)
print(df.head())
