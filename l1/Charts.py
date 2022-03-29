import matplotlib.pyplot as plt

class Charts:
    def __init__(self, title, xlabel, ylabel):
        self.x = list()
        self.y = list()
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.labels = list()

    def loadx(self, lista):
        self.x.append(lista)
        
    def loady(self, lista):
        self.y.append(lista)

    def load(self, x, y, label):
        self.loadx(x)
        self.loady(y)
        self.labels.append(label)
        
    def plot(self):
        for i in range(len(self.x)):
            plt.plot(self.x[i], self.y[i], label=self.labels[i])
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.legend()
        plt.show()

class Splitter():
    def __init__(self, filename):
        self.filename = filename
        self.x = list()
        self.ykrandom = list()
        self.y2opt = list()
        self.ynearest = list()

    def read(self, type):
        file = open(self.filename)
        lines = file.readlines()
        if type == "PRD":
            for line in lines:
                l = line.split(" ")
                self.x.append(int(l[0]))
                self.ykrandom.append(float(l[1]))
                self.y2opt.append(float(l[2]))
                self.ynearest.append(float(l[3]))
        elif type == "memoryBerlin":
            for line in range(len(lines)):
                print(lines[line])
                l = lines[line].split(" ")
                if line < 5:
                    self.x.append(int(l[2]))

                if l[1] == "twoOPT":
                    self.y2opt.append(int(l[3]))
                elif l[1] == "krandom":
                    self.ykrandom.append(int(l[3]))
                elif l[1] == "nearest":
                    self.ynearest.append(int(l[3]))
                else:
                    pass
        file.close()

    def getX(self):
        return self.x

    def getYkrandom(self):
        return self.ykrandom

    def getY2opt(self):
        return self.y2opt

    def getYnearest(self):
        return self.ynearest

# ====================
# == Berlin52Memory
# ====================
#
# s = Splitter("./data/memoryBerlin52.txt")
# s.read("memoryBerlin")
# # print(s.getYnearest())

# char = Charts("Berlin Memory", "k", "Memory (bytes)")
# char.load(s.getX(), s.getYkrandom(), "krandom")
# char.load(s.getX(), s.getY2opt(), "2 OPT")
# char.load(s.getX(), s.getYnearest(), "nearest")
# char.plot()

# ====================
# == PRD
# ====================
#
# s = Splitter("./data/PRD3.txt")
# s.read()

# char = Charts("PRD", "rozmiar instacji", "PRD")
# char.load(s.getX(), s.getYkrandom(), "krandom")
# char.load(s.getX(), s.getY2opt(), "2 OPT")
# char.load(s.getX(), s.getYnearest(), "nearest")
# char.plot()

# ====================
# == PRD - generate data
# ====================
#
# generowanie danych do PRD -> plik ./data/PRD.txt
# for i in range(10, 50, 2):
#     krandomavg, twoOPTabg, nearestavg = list(), list(), list()
#     for j in range(10): # UŚREDNIENIE
#         x = measurePSD(Full(), i)
#         krandomavg.append(x[0])
#         twoOPTabg.append(x[1])
#         nearestavg.append(x[2])
#     print(i, end=" ")
#     print(sum(krandomavg)/len(krandomavg), end=" ")
#     print(sum(twoOPTabg)/len(twoOPTabg), end=" ")
#     print(sum(nearestavg)/len(nearestavg), end=" ")
#     print()