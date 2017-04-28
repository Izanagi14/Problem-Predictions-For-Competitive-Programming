from bs4 import BeautifulSoup
import urllib
import lxml

lis1 = list()
r = urllib.urlopen('http://codeforces.com/submissions/ainta/page/1').read()
ki = 0
a=0
soup = BeautifulSoup(r, 'lxml')
for i in soup.find_all('span',attrs={'class':'page-index'}):
    a = int(i.text)

for kepper in range(1,2):
    r = urllib.urlopen('http://codeforces.com/submissions/ainta/page/' + str(kepper)).read()
    soup = BeautifulSoup(r, 'lxml')
    s = []
    cnt = 1
    #print soup.find_all('td', attrs={'class': 'id-cell'})
    for j in soup.find_all('span', attrs={'class': 'hiddenSource'}):
        #print j.text
        # lis.append(j.text)
        for k in soup.find_all('tr', attrs={'data-submission-id': j.text}):
            x = k.text.split("\n")
            lis1.append((x[12].strip(), x[8].strip(), x[19].strip()))
print lis1
