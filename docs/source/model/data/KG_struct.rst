Knowledge Graph Structure
==========================
This class is responsible for constructing the dataset from the list of triples.
It can also generate substructures and give information about a particular node.

.. autoclass:: model.data.kg_structure.KnowledgeGraph

It has the following methods.

.. autofunction:: model.data.kg_structure.KnowledgeGraph.add_triples
.. autofunction:: model.data.kg_structure.KnowledgeGraph.add
.. autofunction:: model.data.kg_structure.KnowledgeGraph.get_neighbors
.. autofunction:: model.data.kg_structure.KnowledgeGraph.subgraph
.. autofunction:: model.data.kg_structure.KnowledgeGraph.top_entities_sorted_by_conectivity

