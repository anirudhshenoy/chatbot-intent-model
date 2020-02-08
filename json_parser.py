import json
import pickle
import csv
import subprocess as sp

if __name__ == '__main__':

    sp.Popen('npx chatito chatito/chatito-smalltalk --outputPath=\'./chatito\' --trainingFileName=\'smalltalk_dataset.json\'', shell = True).wait()
    sp.Popen('npx chatito chatito/chatito-typeofleave --outputPath=\'./chatito\' --trainingFileName=\'typeofleave_dataset.json\'', shell = True).wait()
    sp.Popen('npx chatito chatito/chatito-main --outputPath=\'./chatito\' --trainingFileName=\'train_dataset.json\'', shell = True).wait()

    files = ['train_dataset.json', 'smalltalk_dataset.json', 'typeofleave_dataset.json']
    filenames = ['chatito_train.csv', 'smalltalk.csv', 'typeofleave.csv']
    for file, filename in zip(files, filenames):
        with open('chatito/' + file) as f:
            data = json.load(f)
        fields = ['data', 'intent']
        lines = []
        entity_data = []
        for intent in data:
            for line in data[intent]:
                sentence = ''
                entity = []
                for value in line:
                    if value['type'] == 'Slot':
                        entity.append((len(sentence), len(sentence) + len(value['value']), value['slot']))
                    sentence += value['value']
                if entity:
                    #print(entity)
                    entity_data.append((sentence, {'entities' : entity}))
                lines.append({'data' : sentence, 'intent' : intent})

        with open('dataset/' + filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = fields)
            writer.writeheader()
            writer.writerows(lines)
        entity_file = open('entity_train_data.pkl', 'wb') 
        pickle.dump(entity_data, entity_file)
        entity_file.close()
        
