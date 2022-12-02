from collections import defaultdict
from random import random

# ------------------ GRAPH DEFINITION ----------------------
class KnowledgeGraph():
    """
    class to generate Knowledge graphs from a set of triples the graph can be directed or non-directed 
    You can choose to add the inverse relations to the representation.

    :param triples: a set of triples (e1, r, e2) to generate the kg.
    :param directed: wether the graph is or not directed. If a directed graph has a connection from e1->r->e2 it does not necessarely have the reverse one.
    :param inverse_triples: wether to populate the graph with the inverse of a triple, i.e. if (e1, r, e2) then (e2, ¬r, e1) (making a directed graph bidirectional) 
    :param verbose: prints information about the progress

    :returns: None
    """
    def __init__(self, triples:list, directed:bool = False, inverse_triples:bool = False, verbose:bool = False):
        self.verbose = verbose
        self.triples = triples
        #{
        # "node1":{"relation1":[node2, node4], "relation2": [node2]}, 
        # "node2":{"relation2":[node5, node7]}
        #}
        self._graph = defaultdict(defaultdict(set).copy)
        self._directed = directed
        self.inverse_triples = inverse_triples

        if(not self._directed and self.inverse_triples):
            self.inverse_triples = False
            print("a non-directed graph, cannot have inverse triples.")
            
        self.add_triples(triples)
    
    def add_triples(self, triples: list):
        """
        Add triples to graph 

        :param triples: a list of triples (e1, r, e2) to generate the kg.

        :returns: None
        """
        for e1, r , e2 in triples:

            self.add(e1, r, e2)
            if(self.verbose):
                print(f"adding triple: {e1}, {r}, {e2}")

            if(self.inverse_triples):
                r_inv = "¬"+r
                self.add(e2, r_inv, e1)
                if(self.verbose):
                    print(f"adding triple: {e2}, ¬{r}, {e1}")
        
    def add(self, e1:str, r:str, e2:str):
        """
        Add connection between e1 and e2 with relation r

        :param e1: the string representation of the first entity in the triple
        :param r: the string representation of the reation in the triple
        :param e2: the string representation of the second entity in the triple

        :returns: None
        """
        self._graph[e1][r].add(e2)
        if(not self._directed):
            self._graph[e2][r].add(e1)
 
    def get_neighbors(self, node:str):
        """
        Returns the neighboring nodes to the requested one

        :param node: the entity to get the inmediate neighbors

        :returns: a dictionary containing the neighbors of the node.
        """
        return self._graph[node]
  
    def subgraph(self, center_node:str, distance:int):
        """
        Given a center node and a neihborhood distance builds a graph of connected entities around the chosen node
        If the distance is 0 it continues until no more nodes are connected.

        :param center_node: the entity to start the subgraph from
        :param distance: depth of subgraph

        :returns: the requested subgraph
        """
        if(distance < 0):
            print("invalid distance")
            raise(IndexError)

        neighbors = self.get_neighbors(center_node)
        values = neighbors.values()
        keys = list(neighbors.keys())

        triples = []
        next_step_entities = []
        next_step_entities_temp = []

        # get the inmediate neighbors and add them to next_step_entities
        # also add their triples to the result set.
        for i, entities in enumerate(values):
            for e in entities:
                triples += [(center_node, keys[i], e)]
                next_step_entities.append(e)

        if distance > 0:
            for _ in range(1, distance):
                for node in next_step_entities:
                    neighbors = self.get_neighbors(node)
                    values = neighbors.values()
                    keys = list(neighbors.keys())

                    for i, entities in enumerate(values):
                        for e in entities:
                            triples += [(node, keys[i], e)]
                            next_step_entities_temp.append(e)

                next_step_entities = next_step_entities_temp
                next_step_entities_temp = []
        
        else:
            visited_nodes = []
            visited_nodes.extend(next_step_entities)

            while len(next_step_entities) > 0:
                #TODO: the way we check for loops is kinda cringe we should make a copy 
                # of the kg with a visitation property in the nodes and update it while we go over it.

                for node in next_step_entities:
                    neighbors = self.get_neighbors(node)
                    values = neighbors.values()
                    keys = list(neighbors.keys())

                    for i, entities in enumerate(values):
                        for e in entities:
                            if(e not in visited_nodes):
                                triples += [(node, keys[i], e)]
                                next_step_entities_temp.append(e)

                next_step_entities = next_step_entities_temp
                next_step_entities_temp = []
        
        res = KnowledgeGraph(triples, directed=True)
        return res
                
    def top_entities_sorted_by_conectivity(self, min_connectivity:int):
        """
        Returns all entities sorted by connectivity that are over or equal the indicated limit.

        :param min_connectivity: the connectivity value to check
        
        :returns: the requested entities
        """
        res = dict()
        for e, r in  self._graph.items():
            tot = 0
            for a in r.values():
                tot+= len(a)

            if(tot >= min_connectivity):
                res[e] = tot

        res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
        return res
    
# t = [("a","b","c"),("c","b","d"),("a","b","d"),("a","f","c"),("d","f","a")]
# kg = KnowledgeGraph(t, directed=True, inverse_triples=True)
# a = kg.get_neighbors("a")
# # neighborhood = kg.subgraph("a", 1)
# neighborhood = kg.subgraph("a", 0)
# print(a)
# print(neighborhood._graph)