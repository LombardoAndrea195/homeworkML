from operator import itemgetter

import jsonlines as js
import csv
def from_jsonl_to_csv2(input_file,file_csv):
    with js.open(input_file) as file:
        with open(file_csv,mode='w') as csv_f:
            fieldnames =['instructions']
            writer=csv.DictWriter(csv_f,fieldnames=fieldnames)
            writer.writeheader()
            righe=file.iter(dict)
            for line in righe:
                string=""
                for ist in line['instructions']:
                    string=string+" "+ist.split()[0]
                writer.writerow({'instructions':string})

def buildthecsv(input_file,file_csv,targetBinario,targetMulticlasse,instruction):
    with js.open(input_file) as file:
        with open(file_csv,mode='w') as csv_f:

            writer = csv.writer(csv_f)
            writer.writerow(('instructions','opt','compiler'))
            for i in range(3000):
                writer.writerow((instruction[i], targetBinario[i],targetMulticlasse[i] ))


def obtainInstructions(input_file,file_csv):
    with js.open(input_file) as file:
        with open(file_csv,mode='w') as csv_f:
            fieldnames =['instructions']
            writer=csv.DictWriter(csv_f,fieldnames=fieldnames)
            writer.writeheader()
            righe=file.iter(dict)
            for line in righe:
                string=""
                for ist in line['instructions']:
                    string=string+" "+ist
                writer.writerow({'instructions':string})