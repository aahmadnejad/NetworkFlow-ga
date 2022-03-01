from random import randint
import numpy as np
from typing import Tuple, List

class GA():
    
    def __init__(self, graph: Tuple[int,int,int], idx: int, in_gens: int, generations: int = 10):
        self.graph = graph
        self.idx = idx
        self.gens = []
        self.in_gens = in_gens
        self.generations = generations
        
        self.run()
        
    def rand_gen(self):
        c=[]
        for item in self.graph: 
            i = (item[0], item[1], randint(0,item[2]))
            c.append(i)
        return c
    
    def first_gen(self):
        for _ in range(self.in_gens):
            c = self.rand_gen()
            self.gens.append((c,self.fitness(c)))
        
    def balance(self, g:Tuple[int,int,int]) -> int:
        matris = np.zeros((self.idx, self.idx))
        for item in g:
            matris[item[0],item[1]] = int(item[2])

        a = np.sum(matris, axis=0) - np.sum(matris, axis=1)
        return len(a) - np.count_nonzero(a)
        
    def to_matris(self, tup: Tuple, idx:int) -> (list, int):
        sum_col = 0
        matris = matris = np.zeros((idx, idx))
        for item in tup:
            matris[item[0],item[1]] = int(item[2])
            sum_col += item[2]
        
        return matris, sum_col
    
    def to_tuple(self, matris: List, idx: int) -> List[Tuple[int,int,int]]:
        res = []
        for i in range(idx):
            for j in range(idx):
               if matris[i][j] != 0:
                   res.append((i,j,int(matris[i][j])))
        return res
    
    def reproduction(self):
            self.gens.sort(key = lambda x: x[1],reverse=True)
            new_chromosome = self.mutation(self.crossover(self.gens[0:2]), self.idx)
            self.gens.append((new_chromosome, self.fitness(new_chromosome)))            
            
    def crossover(self, cr: List[Tuple]):
        chromosomes = []
        for c in cr:
            _2matris, _ = self.to_matris(c[0], self.idx)
            chromosomes.append(_2matris)
        orgMatris, _ = self.to_matris(self.graph, self.idx)
        new_chromosome = np.zeros((self.idx, self.idx))
        
        for i in range(self.idx):
           temp1 = (abs(sum(chromosomes[0][i,:]) - sum(chromosomes[0][:,i])) + 
                    min(sum(orgMatris[i,:]) , sum(orgMatris[:,i])) -
                    max(sum(chromosomes[0][i,:]) , sum(chromosomes[0][:,i])))
           temp2 = (abs(sum(chromosomes[1][i,:]) - sum(chromosomes[1][:,i])) + 
                    min(sum(orgMatris[i,:]) , sum(orgMatris[:,i])) -
                    max(sum(chromosomes[1][i,:]) , sum(chromosomes[1][:,i])))
           
           new_chromosome[i,:] = chromosomes[0][i,:] if temp1 >= temp2 else chromosomes[1][i,:]
           
        return new_chromosome
                
    
    def mutation(self, chromosome: List, idx: int) -> List[Tuple[int,int,int]]:
         col, row = randint(0, idx-1), randint(0, idx-1)
         orgMatris, _ = self.to_matris(self.graph, self.idx)
         for i in range(idx):
             chromosome[col,i] = randint(0, orgMatris[col,i])
             chromosome[i,row] = randint(0, orgMatris[i,row])
         
         return self.to_tuple(chromosome, idx)
    
    def fitness(self, g:Tuple[int,int,int]) -> float:

        orgMatris, orgSum = self.to_matris(self.graph, self.idx)
        matris, gSum = self.to_matris(g, self.idx)

        return ((self.balance(g) / (self.idx-2)) 
                - ((np.sum(matris, axis=0).sum() - np.sum(matris, axis=1).sum()) / gSum )
                + (gSum / orgSum))
    
    def run(self):
        self.first_gen()
        for _ in range(self.generations):
            self.reproduction()
    
    def solution(self):
        print(max(self.gens, key = lambda x: x[1])[0])
        return max(self.gens, key = lambda x: x[1])[0]