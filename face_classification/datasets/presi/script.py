import os 

path1 = "./gray_dataset/"
path2 = "./test/"
b = os.listdir(path1)
c = os.listdir(path2)


for file in b:
    for vid in c:
        if vid[:-4] in file:
            os.remove(str( path1 +file))
            break

print(os.listdir("./T2"))