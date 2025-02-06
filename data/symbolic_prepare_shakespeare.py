import os
import pickle
import numpy as np
from tqdm import tqdm

class OfflineWikidataKnowledgeGraph:
    """
    Interface to an offline Wikidata subset (TSV format).
    """
    def __init__(self, tsv_path):
        self.data = {}  # Store triples: {subject: {predicate: [objects]}}
        self.concept_ids = set() #Keep track of all concepts
        self.relation_ids = set()
        self.load_tsv(tsv_path)


    def load_tsv(self, tsv_path):
        """Loads the Wikidata subset from a TSV file."""
        print(f"Loading Wikidata subset from {tsv_path}...")
        with open(tsv_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header line (subject predicate object)
            for line in tqdm(f, desc="Loading TSV"):
                try:
                    subject, predicate, object_ = line.strip().split('\t')
                    self.concept_ids.add(subject)
                    self.relation_ids.add(predicate)
                    self.concept_ids.add(object_)

                    if subject not in self.data:
                        self.data[subject] = {}
                    if predicate not in self.data[subject]:
                        self.data[subject][predicate] = []
                    self.data[subject][predicate].append(object_)
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}") #Handle lines that do not have 3 values.

        print(f"Loaded {len(self.data)} subjects, {len(self.concept_ids)} unique concepts, and {len(self.relation_ids)} relations.")

    def get_concept_id(self, concept_name):
        """
        Placeholder for getting QID from concept name.  In an offline setting,
        you *must* pre-extract the relevant concepts and their QIDs.  This
        method would look up the QID in a dictionary (created during WDTK processing).

        For this example, we're using the QIDs directly from the TSV.
        """
        return concept_name # Return the QID itself.

    def has_relation(self, concept_id1, concept_id2, relation_id=None):
        """Checks if a relation exists between two concepts."""
        if concept_id1 not in self.data:
            return False
        if relation_id is None:
            # Check for *any* relation
            for rel, objects in self.data[concept_id1].items():
                if concept_id2 in objects:
                    return True
            return False
        else:
            #Check for the specific relation.
            return (relation_id in self.data[concept_id1] and
                    concept_id2 in self.data[concept_id1][relation_id])


    def get_related_concepts(self, concept_id, relation_id=None, limit=None):
        """Gets concepts related to a given concept."""
        related_concepts = []
        if concept_id in self.data:
            if relation_id:
                if relation_id in self.data[concept_id]:
                    related_concepts.extend(self.data[concept_id][relation_id])
            else:
                # Get all related concepts
                for rel, objects in self.data[concept_id].items():
                    related_concepts.extend(objects)
        if limit:
            return related_concepts[:limit]  #Limit output
        else:
            return related_concepts

    def get_relation_id(self, relation_name):
        """
        Placeholder: In a real offline setup with WDTK, you'd pre-extract
        relevant relation IDs (P-numbers) and store them in a dictionary.
        For simplicity, we assume relation IDs are already in the TSV.
        """
        return relation_name


def find_qids(text, kg):
    """
    Finds QIDs in a text using string matching against Wikidata labels.
    This is a *very* basic approach and should be improved.
    """
    qids = set()
    words = text.lower().split() #Lower case
    #Try to find multi-word concepts, from longest to shortest
    for length in range(5, 0, -1):  # Check up to 5-word sequences
      for i in range(len(words) - length + 1):
        phrase = " ".join(words[i:i+length])
        qid = kg.get_concept_id(phrase) # Use the KG
        if qid:
          qids.add(qid)
    return list(qids)


def prepare_symbolic_data(input_file, output_dir, tsv_path, dataset_name="symbolic_shakespeare"):
    """
    Prepares symbolic data from a text file using an offline Wikidata subset.

    Args:
        input_file: Path to the input text file.
        output_dir: Directory to save the processed data.
        tsv_path: Path to the Wikidata subset TSV file.
        dataset_name: name of the dataset
    """

    knowledge_graph = OfflineWikidataKnowledgeGraph(tsv_path)

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # --- Concept and Relation Extraction (using the offline KG) ---
    sentences = text.split('.')  # Very basic sentence splitting
    concept_ids = []
    relation_ids = set()


    print("Extracting concepts and relations...")
    for sentence in tqdm(sentences, desc="Processing sentences"):
        sentence = sentence.strip()
        if not sentence:
            continue

        # 1. Find QIDs in the sentence (very basic string matching)
        current_sentence_qids = find_qids(sentence, knowledge_graph) # Use the helper function

        # 2. Add relations based on Wikidata subset
        for i in range(len(current_sentence_qids) - 1):
            qid1 = current_sentence_qids[i]
            qid2 = current_sentence_qids[i + 1]
            # Check for *any* relation (you can refine this)
            if knowledge_graph.has_relation(qid1, qid2):
                #Find a relation id. In a real scenario, you would iterate and check.
                for rel_id in knowledge_graph.data.get(qid1, {}).keys():
                    if knowledge_graph.has_relation(qid1, qid2, rel_id):
                        relation_ids.add(rel_id) # Add to the set of relation IDs.
                        break #Stop after finding the first relation


        concept_ids.extend(current_sentence_qids)

    print(f"Extracted {len(concept_ids)} concept occurrences and {len(relation_ids)} unique relations.")


    # --- Create Integer Mappings (stoi, itos) ---
    stoi_concepts = {cid: i for i, cid in enumerate(sorted(list(knowledge_graph.concept_ids)))} #Use all known concept ids
    itos_concepts = {i: cid for cid, i in stoi_concepts.items()}

    stoi_relations = {rid: i for i, rid in enumerate(sorted(list(knowledge_graph.relation_ids)))} #And relation ids
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
    # Example usage:
    input_file = 'data/shakespeare/input.txt'  # Path to your text data
    output_dir = 'data/symbolic_shakespeare'   # Output directory
    tsv_path = 'data/wikidata_subset.tsv'      # *** Path to your generated TSV file ***

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    prepare_symbolic_data(input_file, output_dir, tsv_path, dataset_name="symbolic_shakespeare")
