import json
import pickle
import csv

if __name__ == '__main__':

    files = ['training_dataset.json']
    filenames = ['chatito_train.csv']
    for file, filename in zip(files, filenames):
        with open(file) as f:
            data = json.load(f)
        fields = ['data', 'intent']
        #filename = 'chatito_train.csv'

        lines = []
        entity_data = []
        for intent in data:
 #           if intent == 'applyLeave':
            for line in data[intent]:
                sentence = ''
                entity = []
                for value in line:
                    if value['type'] == 'Slot':
                        entity.append((len(sentence), len(sentence) + len(value['value']), value['slot']))
                    sentence += value['value']
                if entity:
                    print(entity)
                    entity_data.append((sentence, {'entities' : entity}))
                lines.append({'data' : sentence, 'intent' : intent})

        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = fields)
            writer.writeheader()
            writer.writerows(lines)
        entity_file = open('entity_train_data.pkl', 'wb') 
        pickle.dump(entity_data, entity_file)
        entity_file.close()
        
