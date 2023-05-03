import json
import gensim
from gensim import corpora
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
import string

# Load JSON
with open('results_nonape.json') as f:
    data = json.load(f)

# Extract summaries
for cluster in range(5):
    summaries = [item['summary'] for item in data if item['cluster_id'] == 1]

    # Define the text cleaning function
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    # Apply the cleaning function to each summary
    clean_summaries = [clean(summary).split() for summary in summaries]

    # Creating the term dictionary of our courpus, where every unique term is assigned an index
    dictionary = corpora.Dictionary(clean_summaries)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above
    doc_term_matrix = [dictionary.doc2bow(summary) for summary in clean_summaries]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Training LDA model on the document term matrix
    ldamodel = Lda(doc_term_matrix, num_topics=15, id2word = dictionary, passes=50)

    # Print the topics
    print(ldamodel.print_topics(num_topics=3, num_words=5))
    print('===============================')
    print("cluster above is :",cluster)
    print('===============================')
    
topics= ["ASR & Multilingual NLP"]
# Load data from JSON file
#with open('papers_summaries.json') as f:
#    data1 = json.load(f)
#
#
#for i in range(20):
#    for example in data1:
#        for cls in data:
#            if cls["cluster_id"] == 0 and cls["paper_id"] == example["paperId"]: 
#                example["focusArea"] = "ASR & Multilingual NLP"
#    
#with open("papers_summaries.json", "w") as f:
#    json.dump(data1, f, indent=4)