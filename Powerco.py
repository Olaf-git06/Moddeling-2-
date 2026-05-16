# -*- coding: utf-8 -*-
"""
Hands-on Linear Optimization
Module 1: Linear Optimization

Powerco Problem
"""
from pulp import *

#Input data
f=open("Powerco.txt","r")
data = f.readlines()
words=data[0].split('\t')

Cities=words[1:len(words)-1]
Parks=[]

Supply=[]
Costs=[]

for i in range(len(data)):
    words=data[i].split('\t')
    if i>0 and i<len(data)-1:
        Costs.append([float(x) for x in words[1:len(words)-1]])
        Supply.append(float(words[len(words)-1]))    
        Parks.append(words[0])            
    elif i==len(data)-1:
        Demand=[float(x) for x in words[1:len(words)]]

NumberParks=len(Parks)
NumberCities=len(Cities)

#create LP problem
Powerco= LpProblem("Powerco", LpMinimize)

#introduce x variable
x = LpVariable.dicts("Electricity",(range(NumberParks),range(NumberCities)),0,None)

#add objective function
Powerco += lpSum([Costs[i][j]*x[i][j] for i in range(NumberParks) for j in range(NumberCities)]), "Total costs"

#constraint supply
for i in range(NumberParks):
    Powerco += lpSum([x[i][j] for j in range(NumberCities)])<=Supply[i]
    
#constraint demand
for j in range(NumberCities):
    Powerco += lpSum([x[i][j] for i in range(NumberParks)])==Demand[j]
 
#solve problem    
Powerco.solve()
    
#print status
print("Status:", LpStatus[Powerco.status])

#print objective function value
print("Transport costs:", value(Powerco.objective)*10000, "euros\n")

#print solution
print("Transported electricity:")
format_row="{:<8}|"+"{:^9}" * NumberCities+"|{:^12}"+"|{:^8}"
print(format_row.format("GWh",*Cities,"Delivered","Supply"))
print('─' * (8+9*NumberCities+12+8+3))
for i in range(NumberParks):
    print(format_row.format(Parks[i],*[value(x[i][j]) for j in range(NumberCities)], sum(value(x[i][j]) for j in range(NumberCities)), Supply[i]))



        
