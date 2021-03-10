import stanza
import time
positive_sentences_id = ""
with open("all.txt",encoding="utf-8") as fp:  
    for cnt, line in enumerate(fp):
        positive_sentences_id = positive_sentences_id+"\n\n"+line

tic = time.perf_counter()
nlp = stanza.Pipeline(lang='en',use_gpu=True,tokenize_pretokenized=True,tokenize_batch_size=9000,lemma_batch_size=9000,pos_batch_size=9000,depparse_batch_size=9000,processors='lemma,tokenize,pos,depparse')
doc = nlp(positive_sentences_id)
with open("sent_output.txt", "w", encoding="utf-8") as f:
        for i, sent in enumerate(doc.sentences):
            #print(i)
            for word in sent.words:
                f.writelines("{:d}\t{:12s}\t{:12s}\t{:6s}\t{:6s}\t{:12s}\t{:d}\t{:12s}\t{:s}\t{:s}".format(\
        word.id, word.text, word.lemma, word.upos, word.xpos, str(word.feats),word.head, word.deprel, "_","_")+"\n")
            #f.writelines("\n")
toc = time.perf_counter()
print(f"The inference time is {toc - tic:0.4f} seconds")
print("end")