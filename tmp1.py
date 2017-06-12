i_am_a_list_of_lists = [["Please don't edit this line"],[],["edit this one: REPLACEME and THISONE"],["need that blank second line"]]

n = 0
mynames = ["this is the replacement"]

for m in mynames:
    print(m)
    print(n)
    mytext = [s.replace('REPLACEME', n.__str__()) for l in i_am_a_list_of_lists for s in l]
    mytext = [s.replace('THISONE', m) for s in mytext]
    print(mytext)
