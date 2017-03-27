def add_new_user(q_for_user):
    import bs4 as bs
    import urllib
    with open("CrawledData.txt","r") as crawleddata:
        names = {}
        questions = []
        for data in (raw.strip().split() for raw in crawleddata):
            questions.append(data[0])
            names[data[0]] = [int(data[1][1])]
            for i in range(2, len(data)):
                names[data[0]].append(int(data[i][0]))
    for i in names:
        print i, names[i]
    for i in names:
        if i in q_for_user['solved']:
            names[i].append(1)
        elif i in q_for_user['todo']:
            names[i].append(0)
        else: 
            names[i].append(3)
    with open("CrawledData.txt","w") as crawleddata:
        for i in names:
            crawleddata.write(i+" ")
            crawleddata.write(str(names[i])+"\n")

    
def get_user_data(user_name):
    import bs4 as bs
    import urllib
    print "Get user data"
    with open("UserName.txt","a+") as users:
        users.write(" "+user_name)
    sauce = urllib.urlopen("http://www.spoj.com/users/"+user_name"/").read()
    soup = bs.BeautifulSoup(sauce, 'lxml')	
    q_for_user = {'solved':[],'todo':[]}
    for i in soup.find_all('table', attrs={'class':'table table-condensed'}):
        q_for_user['solved'] = i.text.strip().split('\n')
        break
    solved = set()
    todo = set()
    for i in (q_for_user['solved']):
        if i != "":
            solved.add(i)
    for i in soup.find_all(lambda tag: tag.name == 'table' and tag.get('class') == ['table']):
        q_for_user['todo'] = i.text.strip().split('\n')
        break
    for i in (q_for_user['todo']):
        if i != "":
            todo.add(i)
    q_for_user['solved'] = list(solved)
    q_for_user['todo'] = list(todo)
    print q_for_user
    return q_for_user

q_for_user = get_user_data('laveshkaushik')
add_new_user(q_for_user)
