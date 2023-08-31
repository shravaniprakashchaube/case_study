# Classifier : Decision tree
# dataset : balls dataset
# features : weight and surface
# labels : Tennis and cricket
# Training Dataset :15 entries
# testing dataset : 1 entry

from sklearn import tree
def Balls(Weight,surface):
    Ballsfeatures = [[35,1],[47,1],[90,1],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0]
                     ,[35,1],[95,0]]
    Names = [1,1,2,1,2,1,2,1,1,1,1,2,1,2,1,2]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Ballsfeatures,Names)
    result = clf.predict([[ Weight,surface]])
    if result == 1:
        print("your object looks like tennis ball")
    elif result == 2:
        print("your object looks like criket ball")


def main():
    print("enter the weight of object")
    Weight = input()
    print("what is the surface type of your object rough or smooth")
    surface = input()
    if surface.lower() == "rough":
        surface = 1
    elif surface.lower() == "smooth":
        surface = 0

    else:
        print("Error: wrong input")
        exit()
    Balls(Weight,surface)

if __name__ == "__main__":
    main()
