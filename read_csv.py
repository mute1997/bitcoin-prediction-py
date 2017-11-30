import csv

def read_csv(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            data.append(row)

    return data
