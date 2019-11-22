import pickle
def storeData(dat, filename):
    dbfile = open(filename, 'ab') 
    pickle.dump(dat, dbfile)                      
    dbfile.close()

def loadData(filename):
    dbfile = open(filename, 'rb')
    return pickle.load(dbfile)

f = open("results.txt", "r")
k = f.readlines()
ans = 0
count = 0
for st in k:
    if st == 'libpng warning: iCCP: known incorrect sRGB profile\n':
        continue
    else:
        try:
            k = int(st)
            ans = ans + k
            count = count + 1
        except:
            if not (count == 0):
                print(ans / count)
            ans = 0
            count = 0