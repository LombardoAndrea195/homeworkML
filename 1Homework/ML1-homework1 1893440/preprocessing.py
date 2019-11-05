import jsonlines as js
import csv
def from_jsonl_to_csv(input_file,file_csv):
    with js.open(input_file) as file:
        with open(file_csv,mode='w') as csv_f:
            fieldnames =['instructions','opt','compiler']
            writer=csv.DictWriter(csv_f,fieldnames=fieldnames)
            writer.writeheader()
            righe=file.iter(dict)
            for line in righe:
                string=""
                for ist in line['instructions']:
                    string=string+" "+ist.split()[0]
                writer.writerow({'instructions':string,'opt':line['opt'],'compiler':line['compiler']})
