from tqdm import tqdm
import rdflib
import sys
import os

class LinkedDataReader():
	"""Reads RDF datasets and provides a .txt file ready for the system."""

	def __init__(self, file_path, prob, include_dataprop, input_format):
		"""
		Arguments:

		file_path -- the path to the single file containing the knwoledge graphs
		prob -- probability of keeping each triple when reading the graph. 
		If 1.0, the entire graph is kept. If lesser than one, the final graph has reduced size.
		"""

		self.file_path = file_path
		self.prob = prob
		self.include_dataprop = include_dataprop
		self.input_format = input_format

	def read(self):
		"""
		Reads the graph using the parameters specified in the constructor.
		Expects each line to contain a triple with the relation first, then the source, then the target.

		Returns: a tuple with:
		1: a dictionary with the entities as keys (their names) as degree information as values.
		Each value is a dictionary with the outwards degree ("out_degree key"), inwards degree ("in_degree key"), total degree ("degree" key), and the data properties ("data_properties" key).
		2: a set with the name of the relations in the graph
		3: a set with the edges in the graph. Each edge is a tuple with the name of the relation, the source entity, and the target entity.
		"""

		entities = dict()
		relations = set()
		edges = set()

		graph = rdflib.Graph()
		graph.parse(self.file_path, rdflib.util.guess_format(self.input_format))
		
		print("Processing statements")
		for source, relation, target in tqdm(graph):
			if(random() < self.prob):
				if source not in entities:
					entities[source] = dict(degree=0, out_degree=0, in_degree=0, data_properties={})
				entities[source]["out_degree"] += 1
				entities[source]["degree"] += 1
				if type(target) is rdflib.term.URIRef:
					if target not in entities:
						entities[target] = dict(degree=0, out_degree=0, in_degree=0, data_properties={})
					entities[target]["in_degree"] += 1
					entities[target]["degree"] += 1
					relations.add(relation)
					edges.add((relation, source, target))
				else:
					if(self.include_dataprop):
						entities[source]["data_properties"][relation] = target

		return (entities, relations, edges)

if __name__ == "__main__":
	assert len(sys.argv) == 2, 'Please provide the RDF file for this dataset as the argument of this script.'
	rdf_dataset_file = sys.argv[1]
	assert os.path.isfile(rdf_dataset_file), 'The specified file does not exist, please try again'

	ldr = LinkedDataReader(rdf_dataset_file, 1, include_dataprop, input_format)

