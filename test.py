class Boutique:
    def __init__(self,bid,bname,btype,brating,points):
        self.bid,self.bname,self.btype,self.brating,self.points=bid,bname,btype,brating,points
class OnlineBoutique:
    def __init__(self,bdict):
        self.bdict=bdict

    def getBoutique(self,lower,upper,extra,typ):
        bt=[]
        bt=self.bdict.get(typ)
        if bt==None:
            return None
        for i in bt:
            if lower<=i.brating<=upper:
                i.points+=extra
        np=[]
        nblist=[]
        for i in bt:
            np.append(i.points)
            np.sort(reverse=True)
        for i in np:
            for j in bt:
                if i==j.points:
                    nblist.append(j)
                return nblist

no=int(input())
blist=[]
for i in range(no):
    bid=int(input())
    bname=input()
    btype=input()
    brating=float(input())
    points=int(input())
    blist.append(Boutique(bid,bname,btype,brating,points))
lr=float(input())
ur=float(input())
extra=int(input())
typ=input()
tlist=[]
for i in blist:
    tlist.append(i.btype)
tlist=set(tlist)
bdict={}
for i in tlist:
    lst=[]
    for j in blist:
        if i==j.btype:
            lst.append(j)
    bdict[i]=lst

tlist=set(tlist)
bdict={}
# t14.$=set(V4A)
# bdict=(}
for i in tlist:
    lst=[]
    for j in blist:
        if i==j.btype:
            lst.append(j)
    bdict[i.lower()]=lst
    lst=[]

myobj=OnlineBoutique(bdict)
ans=myobj.getBoutique(lr,ur,extra,typ.lower())
if ans==None:
    print(lr,ur,extra,typ)
    print("No boutique found")
else:
    for i in ans:
        print(i.bid,i.bname,i.points)
