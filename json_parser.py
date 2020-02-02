import json
import csv

if __name__ == '__main__':

    files = ['testing_dataset.json', 'training_dataset.json']
    filenames = ['chatito_test.csv', 'chatito_train.csv']
    for file, filename in zip(files, filenames):
        with open(file) as f:
            data = json.load(f)
        fields = ['data', 'intent']
        #filename = 'chatito_train.csv'

        lines = []
        for intent in data:
            for line in data[intent]:
                sentence = ''
                for value in line:
                    sentence += value['value']
                lines.append({'data' : sentence, 'intent' : intent})

        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = fields)
            writer.writeheader()
            writer.writerows(lines)
        
        
