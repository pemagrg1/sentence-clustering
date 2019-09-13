import nltk
from SentenceClustering import SentenceClustering



text = """The Russian military has indicated it will supply the Syrian government with a sophisticated air defence system, after condemning a missile attack launched by the US, Britain and France earlier in April. Col Gen Sergei Rudskoi said in a statement on Wednesday that Russia will supply Syria with new missile defence systems soon. Rudskoi did not specify the type of weapons, but his remarks follow reports in the Russian media that Moscow is considering selling its S-300 surface-to-air missile systems to Syria."""
sents = nltk.sent_tokenize(text)
nclusters= 2
sent_clus = SentenceClustering(sents=sents, nclusters=nclusters)
clusters = sent_clus.Agglomerative_clustering()
print (clusters)
for cluster in range(nclusters):
        print ("cluster ",cluster,":")
        for i,sentence in enumerate(clusters[cluster]):
                print ("\tsentence ",i,": ",sents[sentence])