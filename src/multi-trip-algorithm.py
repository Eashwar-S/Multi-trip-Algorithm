# File: multi-trip-algorithm.py
# Author: Eashwar Sathyamurthy
# Date: June 25, 2023
# Description: This script containts implementation of multi-trip algorithm.
# Copyright (c) 2023 Eashwar Sathyamurthy
# MIT License

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import tqdm
import pickle
import copy
plt.rcParams['figure.dpi'] = 300

###########################################
"""
For testing and visualization puroposes.

"""
def createGraph(depotNodes ,requiredEdges, numNodes, show=True):
    G = nx.Graph()
    edges = []
    pos = {}
    reqPos = {}
    s = [1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 10]
    t = [2, 3, 4, 4, 6, 4, 5, 5, 7, 6, 8, 11, 7, 9, 8, 9, 9, 10, 11, 10, 11]
    weights = [2.3, 2, 1.9, 3, 1.5, 3.2, 2.2, 3.8, 2.6, 2.2, 2.8, 2, 1.8, 0.7, 0.8, 0.4, 1.4, 1.5, 0.9, 1.3, 2, 2.5]
    xData = [-2, -0.5, -1,   0, 1,  1.5, 2,   2.5, 3.5, 4.2, 2.7]
    yData = [ 0, -2,    2.5, 0, 3, -2,   0.3, 1.5, -1,  1.2, 3]
    print(len(s), len(t), len(weights))
    for i in range(len(s)):
        edges.append((s[i], t[i], weights[i]))
    
    for i in range(1, numNodes+1):
        G.add_node(i)
        pos[i] =(xData[i-1], yData[i-1])
    
    node_color = ['y']*int(G.number_of_nodes())
    depot_node_color = node_color
    for i in range(1, len(node_color)+1):
        if i in depotNodes:
            depot_node_color[i-1] = 'g'
            
    G.add_weighted_edges_from(edges)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx(G,pos, node_color = node_color)
    nx.draw_networkx(G,pos, node_color = depot_node_color)
    nx.draw_networkx_edges(G, pos, edgelist=requiredEdges, width=3, alpha=0.5,
                                        edge_color="r")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if show:
        plt.figure(1)
        plt.show()
    return G,pos, weights, node_color, depot_node_color

def createGraph1(depotNodes ,requiredEdges, numNodes, show=True):
    G = nx.Graph()
    edges = []
    pos = {}
    reqPos = {}
    s = [1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 10]
    t = [2, 3, 4, 4, 6, 4, 5, 5, 7, 6, 8, 11, 7, 9, 8, 9, 9, 10, 11, 10, 11]
    weights = [2.3, 2, 1.9, 3, 1.5, 2.8, 2.2, 3.8, 2.6, 2.2, 2.8, 2, 1.8, 0.7, 0.8, 0.4, 1.4, 1.5, 0.9, 1.3, 2, 2.5]
    xData = [-2, -0.5, -1,   0, 1,  1.5, 2,   2.5, 3.5, 4.2, 2.7]
    yData = [ 0, -2,    2.5, 0, 3, -2,   0.3, 1.5, -1,  1.2, 3]
    # print(len(s), len(t), len(weights))
    for i in range(len(s)):
        edges.append((s[i], t[i], weights[i]))
    
    for i in range(1, numNodes+1):
        G.add_node(i)
        pos[i] =(xData[i-1], yData[i-1])
    
    node_color = ['y']*int(G.number_of_nodes())
    depot_node_color = node_color
    for i in range(1, len(node_color)+1):
        if i in depotNodes:
            depot_node_color[i-1] = 'g'
            
    G.add_weighted_edges_from(edges)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx(G,pos, node_color = node_color)
    nx.draw_networkx(G,pos, node_color = depot_node_color)
    nx.draw_networkx_edges(G, pos, edgelist=requiredEdges, width=3, alpha=0.5,
                                        edge_color="r")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if show:
        plt.figure(1)
        plt.show()
    return G,pos, weights, node_color, depot_node_color

##############################

def nearestDepot(G, node, depotNodes, radius):
    time = np.inf
    path = []
    dic = {}
    for nodes in depotNodes:
        dic[nodes] = 'yes'
    subG = nx.ego_graph(G, n=node, radius=radius, undirected=False, distance='weight')
    
    for nodes in subG.nodes():
        if dic.get(nodes) != None: 
            t = nx.astar_path_length(G, node, nodes)
            if time > t:
                time = t
                path = nx.astar_path(G, node, nodes)
    return time, path



def feasibleTraversalOfEdge(G, node, edges):
    if node in edges:
        if node == edges[0]:
            return G.get_edge_data(edges[0], edges[1], 0)['weight'], [node, edges[1]]
        else:
            return G.get_edge_data(edges[0], edges[1], 0)['weight'], [node, edges[0]]
    time = np.inf
    path = []
    l1 = nx.astar_path(G, source=node, target=edges[0])
    l2 = nx.astar_path(G, source=node, target=edges[1])

    c1 = nx.astar_path_length(G, source=node, target=edges[0])
    c2 = nx.astar_path_length(G, source=node, target=edges[1])

    if edges[1] != l1[-2]:
        c1 += G.get_edge_data(edges[0], edges[1], 0)['weight']
        l1.append(edges[1])

    if edges[0] != l2[-2]:
        c2 += G.get_edge_data(edges[0], edges[1], 0)['weight']
        l2.append(edges[0])
    # print(l1, l2, c1, c2)
    if c1 <= c2:
        time = c1
        path = l1
    else:
        time = c2
        path = l2
    return time, path



def bestFlight(G, node, Te, untraversedEdges, depotNodes, radius, threshold):
    dic = {}
    pathTime = np.inf
    path = []
    Tstar = np.inf
    Estar = 0
    Qstar = []
    nearestEdgeIndex = -1
    for edges in untraversedEdges:
        dic[str(tuple(edges))] = 'yes'
        
    subG = nx.ego_graph(G, n=node, radius=radius, undirected=False, distance='weight')

    for edges in subG.edges():
        if dic.get(str(edges)) != None or dic.get(str(edges[::-1])) != None:

            if dic.get(str(edges)) != None:
                edgeIndex = untraversedEdges.index(list(edges))
            elif dic.get(str(edges[::-1])) != None:
                edgeIndex = untraversedEdges.index(list(edges[::-1]))

            Te[edgeIndex], Q1 = feasibleTraversalOfEdge(G, node, edges)
            # print('Eashwar')
            # print(Te[edgeIndex], Q1, node, edges)
            timeQ2, Q2 = nearestDepot(G, Q1[-1], depotNodes, radius - Te[edgeIndex])#radius - Te[edgeIndex]

            if Te[edgeIndex] + timeQ2 <= threshold and Te[edgeIndex] < Tstar:
                Tstar = Te[edgeIndex]
                Estar = untraversedEdges[edgeIndex]
                Qstar = Q1
                nearestEdgeIndex = edgeIndex
#             print(Tstar)
    # print(Estar, Tstar, Qstar, nearestEdgeIndex, Te)     
    return Estar, Tstar, Qstar, nearestEdgeIndex, Te



def bestMultiFlight(G, node, Vd, Td, depotNodes, untraversedEdges, radius, threshold, Te):
    dic = {}
    nearestEdge = np.argmin(Te)
    for nodes in depotNodes:
        dic[nodes] = 'yes'
    
    subG = nx.ego_graph(G, n=node, radius=radius, undirected=False, distance='weight')
    
    for nodes in subG.nodes():
        if dic.get(nodes) != None:
            depotIndex = depotNodes.index(nodes)
            Q1 = nx.astar_path(G, source=node, target=nodes)
            Td[depotIndex] = nx.astar_path_length(G, source=node, target=nodes)
            Te_, Q2 = feasibleTraversalOfEdge(G, nodes, untraversedEdges[nearestEdge])
            
            Vd[depotIndex] = Td[depotIndex] + Te_
            
            if Td[depotIndex] > threshold or node == nodes:
                    Vd[depotIndex] = np.inf
    
    return Vd



def multiTripAlgorithm(G, untraversedEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                  uavAvailableTime, uavLastArrivalTimes, uavPaths, uavPathTimes, vehicleCapacity, index):

    multiFlight = False
    numRecharges = 0
    uav_to_requirededges = {}
    start = time.time()
    threshold = len(untraversedEdges)
    pre = len(untraversedEdges)
    # print(G.edges())
    while untraversedEdges:
        
        count = 0
        uav = np.argmin(uavAvailableTime)
    
        Te = np.array([np.inf]*len(untraversedEdges), dtype=np.float32)
        Estar, Tstar, Qstar, nearestEdgeIndex, Te = bestFlight(G, uavLocation[uav], Te, untraversedEdges, depotNodes, vehicleCapacity - uavUtilization[uav], vehicleCapacity - uavUtilization[uav])
        # print(uavPaths, untraversedEdges[nearestEdgeIndex], Estar, uav, Qstar, Tstar)
        if Estar != 0:
            if uavPaths[uav] == 0:
                uavPaths[uav] = [Qstar]
                uavPathTimes[uav] = [Tstar]
            else:
                if multiFlight:
                    if uavLocation[uav] not in depotNodes:
                        uavPaths[uav][-1] += Qstar[1:]
                        uavPathTimes[uav][-1] += Tstar + rechargeTime
                    else:
                        uavPaths[uav].append(Qstar)
                        uavPathTimes[uav].append(Tstar)
                    numRecharges += 1
                elif not multiFlight and uavLocation[uav] in depotNodes:
                    uavPaths[uav].append(Qstar)
                    uavPathTimes[uav].append(Tstar)
                elif not multiFlight and uavLocation[uav] not in depotNodes:
                    uavPaths[uav][-1] += Qstar[1:]
                    uavPathTimes[uav][-1] += Tstar
            if uav_to_requirededges.get(uav) != None:
                    uav_to_requirededges[uav].append(untraversedEdges[nearestEdgeIndex])
            else:
                uav_to_requirededges[uav] = [untraversedEdges[nearestEdgeIndex]]
                    

            uavUtilization[uav] += Tstar
            uavAvailableTime[uav] += Tstar
            # print(multiFlight, uavPaths[uav], uavPaths[uav][-1][-1], Qstar)
            uavLocation[uav] = Qstar[-1]

            if uavLocation[uav] in depotNodes:
                uavLastArrivalTimes[uav] = uavAvailableTime[uav]
                uavAvailableTime[uav] = uavLastArrivalTimes[uav] + rechargeTime
                numRecharges += 1
                uavUtilization[uav] = 0
                multiFlight = False

            untraversedEdges.remove(untraversedEdges[nearestEdgeIndex])

        else:
            Estar = untraversedEdges[np.argmin(Te)]
            Td = [np.inf]*len(depotNodes)
            Vd = np.array([np.inf]*len(depotNodes), dtype=np.float32)
            if uavLocation[uav] in depotNodes:
                multiFlight = True
            else:
                multiFlight = False
            
            
            Vd = bestMultiFlight(G, uavLocation[uav], Vd, Td, depotNodes, untraversedEdges, vehicleCapacity, vehicleCapacity - uavUtilization[uav],  Te) #vehicleCapacity - uavUtilization[uav]
            # print('Eashwar')
            # print(uavLocation[uav], Vd, Td, depotNodes, untraversedEdges, vehicleCapacity, vehicleCapacity - uavUtilization[uav],  Te)
            # print(Vd)
            # print(uav)
            nearestFeasibleDepot = depotNodes[np.argmin(Vd)]
            if Vd[np.argmin(Vd)] < np.inf:
                Q1 = nx.astar_path(G, uavLocation[uav], depotNodes[np.argmin(Vd)])
                TQ1 = nx.astar_path_length(G, uavLocation[uav], depotNodes[np.argmin(Vd)])
                if uavPaths[uav] == 0:
                    uavPaths[uav] = [Q1]
                    uavPathTimes[uav] = [TQ1]
                else:
                    if not multiFlight:
                        uavPaths[uav][-1] += Q1[1:]
                        uavPathTimes[uav][-1] += TQ1
                    else:
                        uavPaths[uav].append(Q1)
                        uavPathTimes[uav].append(TQ1)
                uavLastArrivalTimes[uav] = uavAvailableTime[uav] + nx.astar_path_length(G, uavLocation[uav], depotNodes[np.argmin(Vd)])#depotToDepotDistance[depotNodes.index(uavLocation[uav]), np.argmin(Vd)]

                uavAvailableTime[uav] = uavLastArrivalTimes[uav] + rechargeTime
                numRecharges += 1
                uavLocation[uav] = nearestFeasibleDepot
                uavUtilization[uav] = 0
            else:
                uavAvailableTime[uav] = np.inf
    
    print('Finished Designing Paths')
    print('Completing Incomplete Paths')
    # print(uavPaths)
    # print(uavPathTimes)
    for k in tqdm.tqdm(range(totalUavs)):
        if uavLocation[k] not in depotNodes:
            timeQ1, Q1 = nearestDepot(G, uavLocation[k], depotNodes, vehicleCapacity - uavUtilization[k])
            uavPaths[k][-1] += Q1[1:]
            uavPathTimes[k][-1] += timeQ1
            uavLastArrivalTimes[k] = uavAvailableTime[k] + timeQ1
            uavLocation[k] = Q1[-1]
        uavLastArrivalTimes[k] = round(uavLastArrivalTimes[k], 1)
        if uavPathTimes[k] != 0:
            for i in range(len(uavPathTimes[k])):
                uavPathTimes[k][i] = round(uavPathTimes[k][i], 1)
           
    return uavPaths, uavPathTimes, uavLastArrivalTimes, untraversedEdges, numRecharges, uav_to_requirededges



def visualizePath(G, pos, Node_color, depot_node_color, Edges, depotNodes, requiredNodes, numNodes, path, pathType="solution"):
    for j in range(len(path)):
        if path[j] is not None and path[j] != 'Task Completed':
            # plt.figure(figsize=(10, 10))
    
            G1 = nx.DiGraph()
            pos1 = {}
            node_color = []
            edges = []
            for i in range(len(path[j])-1):
                edges.append((path[j][i], path[j][i+1], G.get_edge_data(path[j][i], path[j][i+1], 0)['weight']))
                pos1[path[j][i]] = pos[path[j][i]]
                if i == len(path[j])-2:
                    pos1[path[j][i+1]] = pos[path[j][i+1]]

            for key in pos1.keys():
                node_color.append(depot_node_color[key-1])

            G1.add_weighted_edges_from(edges)
#             G.add_weighted_edges_from(Edges)
            labels = nx.get_edge_attributes(G,'weight')
            # nx.draw_networkx(G,pos, node_color = Node_color, node_size=20, with_labels=True)
            # nx.draw_networkx(G,pos, node_color = depot_node_color, node_size=20, with_labels=True)
            nx.draw_networkx(G,pos, node_color = Node_color,  with_labels=True)
            nx.draw_networkx(G,pos, node_color = depot_node_color, with_labels=True)
            nx.draw_networkx_edges(G, pos, edgelist=requiredNodes, width=2, alpha=0.5,
                                                edge_color="r")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
            # nx.draw_networkx(G1,pos1, arrows=True, node_color = node_color, edge_color='b', arrowsize=12, width=1, arrowstyle='simple', node_size=20, with_labels=True)
            nx.draw_networkx(G1,pos1, arrows=True, node_color = node_color, edge_color='b', arrowsize=12, width=1, arrowstyle='simple', with_labels=True)
        
            plt.show()
        else:
            continue 

def main(instance_name='dearmon', weather=False):
    if instance_name == 'dearmon':
        position = list(np.load('../dataset/dearmon graph info/pos.npy', allow_pickle=True))
        nodeColor = list(np.load('../dataset/dearmon graph info/node_color.npy', allow_pickle=True))
        depotNodeColor = list(np.load('../dataset/dearmon graph info/depot_node_color.npy', allow_pickle=True))
        Edges = list(np.load('../dataset/dearmon graph info/edges.npy', allow_pickle=True))
        vehCap = list(np.load('../dataset/dearmon graph info/vehicleCapacity.npy', allow_pickle=True))
        nuNod = list(np.load('../dataset/dearmon graph info/numNodes.npy', allow_pickle=True))
        reqNod = list(np.load('../dataset/dearmon graph info/feasibleRequiredNodes.npy', allow_pickle=True))
        deNo = list(np.load('../dataset/dearmon graph info/depotNodes.npy', allow_pickle=True))
        TotalUAVs = list(np.load('../dataset/dearmon graph info/totalUAVs.npy', allow_pickle=True))
        TotalUAVs = list(np.load('../dataset/dearmon graph info/totalUAVs.npy', allow_pickle=True))
        
        instanceData = {'Instance Name' : [],
                        'Number of Nodes' : [],
                        'Number of Edges' : [],
                        'Number of Required Edges' : [],
                        'Capacity' : [],
                        'Total UAVs' : [],
                        'Number of Depot Nodes' : [],
                        'Execution Time' : [],
                        'Maximum Trip Time' : []}
        
        folderPath = '../dataset/dearmon graph files'
        
        print(len(os.listdir(folderPath)))
        for i, file in enumerate(os.listdir(folderPath)):        
            if file.endswith(".net"):
                instanceName = file[0:len(file)-4]
                index = int(file.split('.')[0])
                file_path = f"{folderPath}/{file}"
                # G = nx.read_gpickle(file_path)
                G = nx.read_pajek(file_path)
                mapping = {}
                for node in list(G.nodes()):
                    mapping[node] = int(node)
                G = nx.relabel_nodes(G, mapping)
                pos = position[index]
                node_color = nodeColor[index]
                depot_node_color = depotNodeColor[index]
                edges = Edges[index]
                depotNodes = deNo[index]
                requiredEdges = copy.deepcopy(reqNod[index])
                requiredEdgesCopy = copy.deepcopy(requiredEdges)
                requiredEdgesCopy1 = copy.deepcopy(requiredEdges)
                numRequiredEdges = len(requiredEdges)
                # print(requiredEdges)
                vehicleCapacity = vehCap[index]
                numNodes = nuNod[index]
                totalUavs = TotalUAVs[index]
                uavLocation = []
                numUAVs = {}
                p = 0
                for j in range(totalUavs):
                    if p > len(depotNodes) -1:
                        p = 0
                    if depotNodes[p] in list(numUAVs.keys()):
                        numUAVs[depotNodes[p]] += [depotNodes[p]]
                    else:
                        numUAVs[depotNodes[p]] = [depotNodes[p]]

                    p += 1
                for j in range(len(numUAVs)):
                    uavLocation += numUAVs[depotNodes[j]]

                uavLocation = np.array(uavLocation)
                uavAvailableTime = np.array([0]*totalUavs, dtype=np.float32)
                uavPaths = [0]*totalUavs
                uavPathTimes = [0]*totalUavs
                uavLastArrivalTimes = [0]*totalUavs
                uavUtilization = np.array([0]*totalUavs, dtype=np.float32)
                rechargeTime = 2*vehicleCapacity
                print(rechargeTime)
                instanceData['Instance Name'].append(instanceName)
                instanceData['Number of Nodes'].append(numNodes)
                instanceData['Number of Edges'].append(len(edges))
                instanceData['Number of Required Edges'].append(len(requiredEdges))
                instanceData['Capacity'].append(vehicleCapacity)
                instanceData['Number of Depot Nodes'].append(len(depotNodes))
                instanceData['Total UAVs'].append(totalUavs)

                start = time.time()

                uavPaths, uavPathTimes, uavLastArrivalTimes, traversedEdges, numRecharges, uav_to_requirededges = multiTripAlgorithm(G, requiredEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                                                                            uavAvailableTime, uavLastArrivalTimes, uavPaths, uavPathTimes, vehicleCapacity, index)
                end = time.time()
                
                print('Total UAVs ' + str(totalUavs))
                print(G)
                print('Success rate is ' + str(((numRequiredEdges - len(traversedEdges))/numRequiredEdges)*100) + '%.')
                print('Maximum of all path cost = ' + str(max(uavLastArrivalTimes)))
                print('Total time of all paths = ' + str(sum(uavLastArrivalTimes)))
                print("Execution took "+ str(end-start) + " seconds.")
                print(uavLastArrivalTimes)
                print(uavPaths)
                print(uavPathTimes)


                for uav in range(len(uavPaths)):
                    if uavPaths[uav]:
                        for trip in uavPaths[uav]:
                            # print([uavPaths[uav][i], uavPaths[uav][i+1]])
                            for i in range(len(trip)-1):
                                # print(trip[i], trip[i+1])
                                if requiredEdgesCopy:
                                    for edge in requiredEdgesCopy:
                                        if edge == [trip[i], trip[i+1]] or edge[::-1] == [trip[i], trip[i+1]]:
                                            requiredEdgesCopy.remove(edge)

                instanceData['Execution Time'].append(round(end-start, 3))
                instanceData['Maximum Trip Time'].append(max(uavLastArrivalTimes))
                
                if not requiredEdgesCopy:
                    print('Repaired Trips are correct')
                    print(instanceData)
                    df = pd.DataFrame(instanceData)
                    df.to_csv('../results/dearmon_instances_results.csv')

    elif instance_name == 'real-world':
        position = list(np.load('../dataset/real world graph info/pos.npy', allow_pickle=True))
        nodeColor = list(np.load('../dataset/real world graph info/node_color.npy', allow_pickle=True))
        depotNodeColor = list(np.load('../dataset/real world graph info/depot_node_color.npy', allow_pickle=True))
        Edges = list(np.load('../dataset/real world graph info/edges.npy', allow_pickle=True))
        vehCap = list(np.load('../dataset/real world graph info/vehicleCapacity.npy', allow_pickle=True))
        nuNod = list(np.load('../dataset/real world graph info/numNodes.npy', allow_pickle=True))
        reqNod = list(np.load('../dataset/real world graph info/feasibleRequiredNodes1.npy', allow_pickle=True))
        deNo = list(np.load('../dataset/real world graph info/depotNodes.npy', allow_pickle=True))
        depotNodesLatLong = list(np.load('../dataset/real world graph info/depotNodesLatLong.npy', allow_pickle=True))
        normalNodesLatLong = list(np.load('../dataset/real world graph info/normalNodesLatLong.npy', allow_pickle=True))
        requiredEdgesID = list(np.load('../dataset/real world graph info/feasibleRequiredEdgesID.npy', allow_pickle=True))
        nonrequiredEdgesID = list(np.load('../dataset/real world graph info/nonrequiredEdgesID.npy', allow_pickle=True))
        dic = list(np.load('../dataset/real world graph info/dic.npy', allow_pickle=True))
        timeStamp = list(np.load('../dataset/real world graph info/timeStamp.npy', allow_pickle=True))
        dicMapList = list(np.load('../dataset/real world graph info/dicMapList.npy', allow_pickle=True))
    #     info = readAndStoreInstanceInfo('../dataset/CARP_datasets/DeArmon_gdb-IF')
        instanceData = {'Instance Name' : [],
                        'Number of Nodes' : [],
                        'Number of Edges' : [],
                        'Number of Required Edges' : [],
                        'Capacity' : [],
                        'Total UAVs' : [],
                        'Number of Depot Nodes' : [],
                        'Time Stamp' : [],
                        'Execution Time' : [],
                        'Maximum Routes Time' : [],
                        'Total Route Times' : [],
                        'Success Rate' : [],
                        'Number of Recharges' : []}
        if weather:
            folderPath = '../dataset/real world graph files/icy road weather instance graph/'
        else:
            folderPath = '../dataset/real world graph files/new icy road instance graph/'
        actuaLGraphPath = '../dataset/real world graph files/new graph/'
        unfilteredGraphPath = '../dataset/real world graph files/unfiltered new graph/'
        for i, file in enumerate(tqdm.tqdm(os.listdir(folderPath), position=0)):
            if file.endswith(".pkl"):
                index = int(file.split('.')[0].split('US')[-1])
                instanceName = file[0:len(file)-4]
                file_path = f"{folderPath}/{file}"
                actual_file_path = f"{actuaLGraphPath}/{file}"
                unfiltered_file_path = f"{unfilteredGraphPath}/{file}"
                with open(actual_file_path, 'rb') as f:
                    graph = pickle.load(f)
                # nx.write_pajek(graph, f"{actuaLGraphPath}/{file1}")
                with open(unfiltered_file_path, 'rb') as f:
                    Ugraph = pickle.load(f)
                # nx.write_pajek(Ugraph, f"{unfilteredGraphPath}/{file1}")
                with open(file_path, 'rb') as f:
                    G = pickle.load(f)
                pos = position[index]
                node_color = nodeColor[index]
                depot_node_color = depotNodeColor[index]
                edges = Edges[index]
                depotNodes = deNo[index]
                requiredEdges = reqNod[index]
                requiredEdgesCopy = requiredEdges.copy()
                numRequiredEdges = len(requiredEdges)

                print(G)
                print('numReqEdges :' + str(numRequiredEdges))
                vehicleCapacity = 31

                numNodes = nuNod[index]
                totalUavs = round(len(requiredEdges)//2)
                uavLocation = []
                numUAVs = {}
                p = 0
                for j in range(totalUavs):
                    if p > len(depotNodes) -1:
                        p = 0
                    if depotNodes[p] in list(numUAVs.keys()):
                        numUAVs[depotNodes[p]] += [depotNodes[p]]
                    else:
                        numUAVs[depotNodes[p]] = [depotNodes[p]]

                    p += 1
                for j in range(len(numUAVs)):
                    uavLocation += numUAVs[depotNodes[j]]
                uavLocation = np.array(uavLocation)
                uavAvailableTime = np.array([0]*totalUavs, dtype=np.float32)
                uavPaths = [0]*totalUavs
                uavPathTimes = [0]*totalUavs
                uavLastArrivalTimes = [0]*totalUavs
                uavUtilization = np.array([0]*totalUavs, dtype=np.float32)
                rechargeTime = 90

                instanceData['Instance Name'].append(instanceName)
                instanceData['Number of Nodes'].append(numNodes)
                instanceData['Number of Edges'].append(G.number_of_edges())
                instanceData['Number of Required Edges'].append(len(requiredEdges))
                instanceData['Capacity'].append(vehicleCapacity)
                instanceData['Number of Depot Nodes'].append(len(depotNodes))
                instanceData['Total UAVs'].append(totalUavs)
                instanceData['Time Stamp'].append(timeStamp[index])
                start = time.time()

    #                 preprocessing(G, depotNodes, requiredEdges) 
                uavPaths, uavPathTimes, uavLastArrivalTimes, traversedEdges, numRecharges, uav_to_requirededges = multiTripAlgorithm(G, requiredEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                                                                            uavAvailableTime, uavLastArrivalTimes, uavPaths, uavPathTimes, vehicleCapacity, index)

                end = time.time()
                allRoutes = []
                for k in range(totalUavs):
                    if uavPaths[k] != 0:
                        for paths in uavPaths[k]:
                            allRoutes.append(paths)
    #                     allRoutes.append(routes)
                print('Success rate is ' + str(((numRequiredEdges - len(traversedEdges))/numRequiredEdges)*100) + '%.')

    

                print('Maximum of all path cost = ' + str(max(uavLastArrivalTimes)))
        #                 print('Number of times drones had to recharge = ' + str(numRecharge))
                print('Total time of all paths = ' + str(sum(uavLastArrivalTimes)))
                print("Execution took "+ str(end-start) + " seconds.")
                instanceData['Execution Time'].append(round(end-start, 3))
                instanceData['Maximum Routes Time'].append(max(uavLastArrivalTimes))
                instanceData['Total Route Times'].append(sum(uavLastArrivalTimes))
                instanceData['Number of Recharges'].append(numRecharges)
                instanceData['Success Rate'].append(((numRequiredEdges - len(traversedEdges))/numRequiredEdges)*100)
                print()
                print()
                ## Save the results in .csv format
                df = pd.DataFrame(instanceData)
                if weather:
                    df.to_csv('../results/real_world_weather_instances_results.csv')
                else:
                    df.to_csv('../results/real_world_instances_results.csv')

if __name__ == "__main__":
    main(instance_name='dearmon')       # For running dearmon instances
    main(instance_name='real-world', weather=False) # For running real-world road networks without considering wind directions
    main(instance_name='real-world', weather=True)  # With considering wind directions

    
# For testing

# # # inputs
if __name__ == "__main__":
    ## Test-case 1
#     vehicleCapacity = 7
#     rechargeTime = 1.1
#     # rechargeTime = 2*vehicleCapacity
#     numNodes = 11
#     requiredEdges = [[4, 5], [2, 6], [7, 8], [10, 11]]
#     requiredEdgesCopy = copy.deepcopy(requiredEdges)
#     uavLocation = np.array([1, 9])
#     totalUavs = uavLocation.size
#     uavAvailableTime = np.array([0]*totalUavs, dtype=np.float32)
#     uavPaths = [0]*totalUavs
#     uavPathTimes = [0]*totalUavs
#     uavLastArrivalTimes = [0]*totalUavs
#     # uavLocation = np.array([1, 5])
#     uavUtilization = np.array([0]*totalUavs, dtype=np.float32)
#     numrequiredEdges = len(requiredEdges)
#     depotNodes = [1, 5, 9]
#     G,pos, weights, Node_color, depot_node_color = createGraph1(depotNodes, requiredEdges, numNodes)

    ## Test-case 2
    vehicleCapacity = 7
    rechargeTime = 1.1
    numNodes = 11
    requiredEdges = [[3, 4], [4, 7], [2, 6], [7, 8], [10, 11]]
    requiredEdgesCopy = copy.deepcopy(requiredEdges)
    uavLocation = np.array([1, 5, 9])
    totalUavs = uavLocation.size
    uavAvailableTime = np.array([0]*totalUavs, dtype=np.float32)
    uavPaths = [0]*totalUavs
    uavPathTimes = [0]*totalUavs
    uavLastArrivalTimes = [0]*totalUavs
    # uavLocation = np.array([1, 5])
    uavUtilization = np.array([0]*totalUavs, dtype=np.float32)
    numrequiredEdges = len(requiredEdges)
    depotNodes = [1, 5, 9]
    print(requiredEdges)
    G,pos, weights, Node_color, depot_node_color = createGraph1(depotNodes, requiredEdges, numNodes)

    ## Test-case 3
#     vehicleCapacity = 7
#     rechargeTime = 1.1
#     # rechargeTime = 2*vehicleCapacity
#     numNodes = 11
#     requiredEdges = [[3, 4], [7, 8], [10, 11]]
#     requiredEdgesCopy = copy.deepcopy(requiredEdges)
#     uavLocation = np.array([1, 9])
#     totalUavs = uavLocation.size
#     uavAvailableTime = np.array([0]*totalUavs, dtype=np.float32)
#     uavPaths = [0]*totalUavs
#     uavPathTimes = [0]*totalUavs
#     uavLastArrivalTimes = [0]*totalUavs
#     # uavLocation = np.array([1, 5])
#     uavUtilization = np.array([0]*totalUavs, dtype=np.float32)
#     numrequiredEdges = len(requiredEdges)
#     depotNodes = [1, 9]
#      G,pos, weights, Node_color, depot_node_color = createGraph(depotNodes, requiredEdges, numNodes)


    uavPaths, uavPathTimes, uavLastArrivalTimes, edges1, numRecharges, uav_to_requirededges = multiTripAlgorithm(G, requiredEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                uavAvailableTime, uavLastArrivalTimes, uavPaths, uavPathTimes, vehicleCapacity, index=0)
    print(uavPaths)
    print(uavPathTimes)
    print(uavLastArrivalTimes)

