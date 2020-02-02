import pickle 
import random
from spacy.util import minibatch, compounding
import spacy



def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.load('en_core_web_sm')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)
    return nlp


if __name__ == '__main__':
    dbfile = open('entity_train_data.pkl', 'rb')      
    db = pickle.load(dbfile) 
    print(db[0])

    prdnlp = train_spacy(db, 8)

    # Save our trained Model
    prdnlp.to_disk('custom_ner')

    while True:
        test_text = input("Enter your testing text: ")
        doc = prdnlp(test_text)
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)