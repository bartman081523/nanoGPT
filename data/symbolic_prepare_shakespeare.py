import os
import pickle
import requests
import numpy as np
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON

class WikidataKnowledgeGraph:
    """
    Interface to Wikidata using SPARQL.
    """
    def __init__(self, endpoint="https://query.wikidata.org/sparql"):
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)

    def get_concept_id(self, concept_name):
        """
        Retrieves the Wikidata QID for a given concept name (label).

        Args:
            concept_name: The string label of the concept (e.g., "apple").

        Returns:
            The Wikidata QID (e.g., "Q89") as a string, or None if not found.
        """
        query = f"""
        SELECT ?item ?itemLabel
        WHERE {{
          ?item rdfs:label "{concept_name}"@en .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        LIMIT 1
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.queryAndConvert()
            for result in results["results"]["bindings"]:
                return result["item"]["value"].split("/")[-1]  # Extract QID
            return None  # No results found
        except Exception as e:
            print(f"Error during SPARQL query: {e}")
            return None  # Handle query errors gracefully

    def get_relation_id(self, relation_name):
      """
      Retrieves the Wikidata PID for a given relation name
      """
      query = f"""
        SELECT ?item ?itemLabel
        WHERE {{
          ?item rdfs:label "{relation_name}"@en .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        LIMIT 1
        """
      self.sparql.setQuery(query)
      try:
          results = self.sparql.queryAndConvert()
          for result in results["results"]["bindings"]:
              return result["item"]["value"].split("/")[-1]  # Extract PID
          return None
      except Exception as e:
        print(f"SPARQL query error: {e}")
        return None


    def has_relation(self, concept_id1, concept_id2, relation_id=None):
        """
        Checks if a relation exists between two concepts in Wikidata.

        Args:
            concept_id1: The Wikidata QID of the first concept.
            concept_id2: The Wikidata QID of the second concept.
            relation_id:  The Wikidata PID of the relation (Property ID).  If None, *any* relation is checked.

        Returns:
            True if the relation exists, False otherwise.
        """

        if relation_id:
             query = f"""
                ASK {{
                  wd:{concept_id1} wdt:{relation_id} wd:{concept_id2} .
                }}
            """
        else:
            # Check for *any* relation
            query = f"""
               ASK {{
                  wd:{concept_id1} ?p wd:{concept_id2} .
                  FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
                }}
            """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.queryAndConvert()
            return results["boolean"]
        except Exception as e:
            print(f"Error during SPARQL query: {e}")
            return False  # Handle errors: assume no relation

    def get_related_concepts(self, concept_id, relation_id=None, limit=5):
        """
        Gets concepts related to a given concept via a specific relation or any relation.

        Args:
            concept_id: The Wikidata QID of the concept.
            relation_id: (Optional) The Wikidata PID of the relation. If None, all relations are considered.
            limit: Maximum number of related concepts to return.
        Returns:
             A list of related concept QIDs.
        """

        if relation_id:
            query = f"""
                SELECT ?related ?relatedLabel
                WHERE {{
                  wd:{concept_id} wdt:{relation_id} ?related .
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                }}
                LIMIT {limit}
                """
        else:
            query = f"""
                SELECT ?related ?relatedLabel ?rel ?relLabel
                WHERE {{
                  wd:{concept_id} ?rel ?related.
                  FILTER(STRSTARTS(STR(?rel), "http://www.wikidata.org/prop/direct/"))
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                }}
                LIMIT {limit}
                """
        self.sparql.setQuery(query)
        related_concepts = []
        try:
            results = self.sparql.queryAndConvert()
            for result in results["results"]["bindings"]:
                related_concepts.append(result["related"]["value"].split("/")[-1])
            return related_concepts
        except Exception as e:
            print(f"Error during SPARQL: {e}")
            return []

def prepare_symbolic_data(input_file, output_dir, dataset_name="symbolic_data"):
    """
    Prepares symbolic data from a text file using Wikidata.

    Args:
        input_file: Path to the input text file.
        output_dir: Directory to save the processed data.
        dataset_name: name of the dataset (train and val split).
    """
    knowledge_graph = WikidataKnowledgeGraph()

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # --- Concept and Relation Extraction (using Wikidata) ---
    sentences = text.split('.')  # Very basic sentence splitting.
    concept_ids = []
    relation_ids = set()
    all_concepts = set()  #Keep track of all concepts for creating stoi/itos
    all_relations = set()

    print("Extracting concepts and relations...")
    for sentence in tqdm(sentences, desc="Processing sentences"):
        words = sentence.strip().split()  # Basic tokenization
        current_sentence_concepts = []

        for word in words:
            concept_id = knowledge_graph.get_concept_id(word.lower())
            if concept_id:
                current_sentence_concepts.append(concept_id)
                all_concepts.add(concept_id) #Add to overall concept list

        # Simple relation extraction (look for adjacent concepts in Wikidata)
        # This part is crucial and can be significantly improved using more context
        for i in range(len(current_sentence_concepts) - 1):
            concept1 = current_sentence_concepts[i]
            concept2 = current_sentence_concepts[i + 1]
            for rel_name in ["part of", "has part", "instance of", "subclass of", "located in", "member of"]: #Examples
               rel_id = knowledge_graph.get_relation_id(rel_name)
               if rel_id:
                  all_relations.add(rel_id)
                  if knowledge_graph.has_relation(concept1, concept2, rel_id):
                        # Collect relations, if needed.
                        relation_ids.add(rel_id)
                        break #Stop at the first found relation

        concept_ids.extend(current_sentence_concepts)  # Add concepts from the current sentence.

    print(f"Extracted {len(concept_ids)} concept occurrences, {len(all_concepts)} unique concepts, and {len(relation_ids)} unique relations.")

     # --- Create Integer Mappings (stoi, itos) ---
    stoi_concepts = {cid: i for i, cid in enumerate(sorted(list(all_concepts)))}
    itos_concepts = {i: cid for cid, i in stoi_concepts.items()}

    stoi_relations = {rid: i for i, rid in enumerate(sorted(list(all_relations)))}
    itos_relations = {i: rid for rid, i in stoi_relations.items()}

    # --- Encode Data ---
    encoded_concept_ids = [stoi_concepts[cid] for cid in concept_ids]

    # --- Split into Train/Val ---
    n = len(encoded_concept_ids)
    train_data = encoded_concept_ids[:int(n*0.9)]
    val_data = encoded_concept_ids[int(n*0.9):]

    # --- Export to Binary Files ---
    train_ids = np.array(train_data, dtype=np.uint16)  # Or uint32 if needed
    val_ids = np.array(val_data, dtype=np.uint16)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    # --- Save Metadata ---
    meta = {
        'vocab_size': len(stoi_concepts),  # Concept vocab size
        'stoi': stoi_concepts,
        'itos': itos_concepts,
        'relation_stoi': stoi_relations,
        'relation_itos': itos_relations,
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved metadata to {os.path.join(output_dir, 'meta.pkl')}")



if __name__ == '__main__':
    # Example usage (assuming you have an 'input.txt' file):
    input_file = 'data/shakespeare/input.txt' # Use your data
    output_dir = 'data/symbolic_shakespeare' #Example

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    prepare_symbolic_data(input_file, output_dir, dataset_name = "symbolic_shakespeare")
