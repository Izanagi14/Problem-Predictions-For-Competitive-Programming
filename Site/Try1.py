import csv

from final_data import dataset2

lis = list()
for k, v in dataset2.items():
    lis.append(k)
with open(("../collect.csv"), 'wb') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["User", "Problem", "Rating"], delimiter=';')
    writer.writeheader()
with open(("../collect.csv"), 'wb') as csvfile:
    for i in lis:
        print i
        # print dataset2[i]
        lis2 = list()
        for k, v in dataset2[i].items():
            csvfile.write(k)