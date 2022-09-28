from collections import defaultdict
from random import random

# ------------------ GRAPH DEFINITION ----------------------
class KnowledgeGraph():
    """
    Class to generate Knowledge graphs from a set of triples
    the graph can be directed or non-directed 
    You can choose to add the inverse relations to the representation.
    """
    def __init__(self, triples, directed = False, inverse_triples = False, verbose = False):
        self.verbose = verbose
        self._triples = triples
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
    
    def add_triples(self, triples):
        """ Add triples (list of triples) to graph """
        for e1, r , e2, *result in triples:

            self.add(e1, r, e2)
            if(self.verbose):
                print(f"adding triple: {e1}, {r}, {e2}")

            if(self.inverse_triples):
                r_inv = "¬"+r
                self.add(e2, r_inv, e1)
                if(self.verbose):
                    print(f"adding triple: {e2}, ¬{r}, {e1}")
        
    def add(self, e1, r, e2):
        """ Add connection between e1 and e2 with relation r """
        self._graph[e1][r].add(e2)
        if(not self._directed):
            self._graph[e2][r].add(e1)
 
    def get_neighbors(self, node):
        '''Returns the neighboring nodes to the requested one'''
        return self._graph[node]
  
    def subgraph(self, center_node, distance):
        '''
        Given a center node and a neihborhood distance builds a graph of connected entities around the chosen node
        If the distance is 0 it continues until no more nodes are connected.
        '''
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
                
    def remove_random_subset(self, connectivity = 3, removal_percentage = 10):
        '''
        Removes a number of triples from the graph that correspond to the indicated removal percentage

        connectivity: the minimum number of connected nodes so that te entity could be selecte for triple removal.
        
        removal_percentage: the percentage of the graph that wants to be removed from the selected triples, 
        if the % is over the total graph size the returned value will be None, recommended value = 10

        the removal process will avoid completely removing an entity from the graph.

        ATTENTION: THIS IS A VERY HEAVY OPERATION.
        '''
        top = self.top_entities_sorted_by_conectivity(connectivity)
        top_sum = sum(top.values())-len(top.values()) # we don't want to remove all the conectivity from nodes so we really have 1 less per triple.

        n_percent_triples = len(self._triples) * (removal_percentage/100)

        if (n_percent_triples < top_sum):
            print(f"surpassed threshold {int(n_percent_triples)} of graph, removing random subset from the top --> {top_sum} triples.")
            removed_triples = []
            
            for _ in range(int(n_percent_triples)):
                parent_entity, relation, tail_entity = self.recursive_triple_removal(top)
                removed_triples.append((parent_entity, relation, tail_entity))

            # self._triples = [x for x in self._triples if x not in removed_triples]
            return removed_triples

        else:
            raise ValueError(f"threshold not met {int(n_percent_triples)} of graph to be removed, connectivity subset is --> {top_sum} triples only.")

    def top_entities_sorted_by_conectivity(self, min_connectivity):
        '''
        Returns all entities sorted by connectivity that are over or equal the indicated limit.
        '''
        res = dict()
        for e, r in  self._graph.items():
            tot = 0
            for a in r.values():
                tot+= len(a)

            if(tot >= min_connectivity):
                res[e] = tot

        res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
        return res

    def recursive_triple_removal(self, top):
        '''
        Removes an entity from the graph taking into account that it cannot fully disconnect it.
        Calls itself in case it would to check for a new one.
        '''
        parent_entity = random.choices(list(top.keys()), top.values())[0]
        top[parent_entity] -= 1
        triples_dict = self._graph[parent_entity]

        relations = list(triples_dict.keys())
        len_ent_relations = []

        for v in triples_dict.values():
            len_ent_relations.append(len(v))

        if(sum(len_ent_relations) == 1):
            #removing this triple would disconnect the entity from the graph as a parent entity.
            print(f"this removal would disconnect ({parent_entity}) from graph, retrying...")
            return self.recursive_triple_removal(top) # so we try again.

        relation = random.choices(relations, len_ent_relations)[0]

        relation_index = relations.index(relation)
        aux = list(triples_dict.values())

        tail_entity = random.choice(list(aux[relation_index]))
        # print(f"{parent_entity}, {relation}, {tail_entity}")


        self.remove_triple(parent_entity, relation, tail_entity)
        return parent_entity, relation, tail_entity

    
# t = [("a","b","c"),("c","b","d"),("a","b","d"),("a","f","c"),("d","f","a")]
# kg = KnowledgeGraph(t, directed=True, inverse_triples=True)
# a = kg.get_neighbors("a")
# # neighborhood = kg.subgraph("a", 1)
# neighborhood = kg.subgraph("a", 0)
# print(a)
# print(neighborhood._graph)