import imp


from Charts import Charts

xAxis = []
yAxis = []
lastline = None
c = 0

f = open("wykresZterminalaDane.txt", "r")
for line in f.readlines():
    line = line.split(" ")
    line[2] = line[2].split("\n")[0]


    print(line)
    if c % 50 == 0:
        xAxis.append(int(line[0]))
        yAxis.append(float(line[2]))
    lastline = line
    c += 1


xAxis.append(int(lastline[0]))
yAxis.append(float(lastline[2]))
compareLine = [23 for i in xAxis]

# yAxis = yAxis[::-1]

chart = Charts("Ilośc iteracji PRD dla gr120", "iteracja", "PRD")
chart.load(xAxis, yAxis, "blue", "PRD")
chart.load(xAxis, compareLine, "red", "nearest_neighbour")
chart.plot()

f.close()