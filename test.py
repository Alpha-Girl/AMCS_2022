from locale import normalize
import networkx as nx
 
G = nx.Graph()
 
# 从文件@filename中读取网络的adjacentMatrix，通过networkx的add_edges方法向对象G中添加边
def readNetwork(filename):
	fin = open(filename, 'r')
	# for line in fin:
	# 	for node in line:
	# 		print(node, end="")
			
	# lines = fin.readlines()
	# print(len(lines))
 
	rowCount = 1
	colCount = 1
	for line in fin.readlines():
		line = line.split(" ")
		for node in line:
			if node == '1':
				G.add_edge(rowCount, colCount)
			colCount = colCount + 1
		colCount = 1
		rowCount += 1
 
	print(G.edges())
 
# 计算网络中的节点的介数中心性，并进行排序输出
def topNBetweeness():
	score = nx.betweenness_centrality(G,normalized=False)
	score = sorted(score.items(), key=lambda item:item[1], reverse = True)
	print("betweenness_centrality: ", score)
	output = []
	for node in score:
		output.append(node[0])
 
	print(output)
	fout = open("D:\\daquan\\result.data", 'w')
	for target in output:
		fout.write(str(target)+" ")
 
readNetwork("D:\\daquan\\betweennessSorted.data")
topNBetweeness()