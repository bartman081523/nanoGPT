import os
import pickle
import subprocess  # Import subprocess

import spacy
import numpy as np
from tqdm import tqdm
from spacy.tokens import Span


class OfflineWikidataKnowledgeGraph:
    """
    (Same OfflineWikidataKnowledgeGraph class as before - no changes here)
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
        Placeholder: Returns the QID. In this offline setting,
        we are using the QIDs directly from the TSV.
        """
        return concept_name  # Return the QID itself

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
            return related_concepts[:limit]
        else:
            return related_concepts

    def get_relation_id(self, relation_name):
        """
        Placeholder
        """
        return relation_name


def create_wikidata_subset(dump_file, output_tsv, qid_file, java_jar_path):
    """
    Creates the Wikidata subset using the compiled Java program.

    Args:
        dump_file: Path to the Wikidata dump file (e.g., .json.bz2).
        output_tsv: Path to the output TSV file.
        qid_file: Path to the file containing initial QIDs (one per line).
        java_jar_path: Path to the compiled Java JAR file.
    """

    command = [
        "java",
        "-jar",
        java_jar_path,
        dump_file,
        output_tsv,
        qid_file,
    ]
    print(f"Running Java command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)  # Print standard output from Java
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running Java command: {e}")
        print(e.stderr)
        exit(1) #Exit with an error, if Java programm could not run
    except FileNotFoundError:
        print("Error: Java not found.  Make sure Java is installed and in your PATH.")
        exit(1)


def prepare_symbolic_data(input_file, output_dir, tsv_path, wikidata_dump_path, dataset_name="symbolic_shakespeare"):
    """
    Prepares symbolic data, creating a Wikidata subset if needed.
    """

    qid_file = os.path.join(output_dir, "initial_qids.txt") #Temporary QID file
    java_jar_path = os.path.join("wikidata", "wikidata-subset-creator.jar") # Relative Path

    # --- 0. Create Wikidata Subset (if it doesn't exist) ---
    if not os.path.exists(tsv_path):
        print(f"Wikidata subset file not found: {tsv_path}")
        # Very basic initial QID extraction using spaCy (before KG subset exists)
        #We extract QIDs from the input file, and add them to the initial QID file.
        print("Performing initial entity linking with spaCy to generate qids.txt...")
        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_lg")
            nlp = spacy.load("en_core_web_lg")
        if "entityLinker" not in nlp.pipe_names and hasattr(nlp, "add_pipe"):
             nlp.add_pipe("entityLinker", last=True)

        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        doc = nlp(text)
        initial_qids = set()
        for ent in doc.ents:
            if ent._.kb_ents:
                initial_qids.add(ent._.kb_ents[0][0])
        with open(qid_file, "w", encoding="utf-8") as f:
            for qid in initial_qids:
                f.write(qid + "\n")
        print(f"Initial QIDs written to {qid_file}")
        create_wikidata_subset(wikidata_dump_path, tsv_path, qid_file, java_jar_path)  # Create the subset


    # --- 1. Load Offline Wikidata Knowledge Graph ---
    knowledge_graph = OfflineWikidataKnowledgeGraph(tsv_path)

    # --- 2. Load spaCy Model (and download if necessary) ---
    try:
        nlp = spacy.load("en_core_web_lg")  # Use a larger model
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
    if "entityLinker" not in nlp.pipe_names and hasattr(nlp, "add_pipe"):
       nlp.add_pipe("entityLinker", last=True)

    # --- 3. Load and Process Text ---
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # --- 4. Entity Linking with spaCy ---
    print("Performing entity linking with spaCy...")
    doc = nlp(text)
    concept_ids = []  # Collect QIDs (as strings)
    relation_ids = set() #Collect Relation IDs

    linked_qids = set() #Keep track of linked QIDs.

    for ent in doc.ents:
        if ent._.kb_ents:
            q_id = ent._.kb_ents[0][0]  # Get top candidate QID
            linked_qids.add(q_id)
            # concept_ids.append(q_id) # Don't add here, add after relation extraction

    # --- 5. Relation Extraction (using KG + adjacent linked entities) ---
    print("Extracting relations...")

    sentences = text.split('.')
    for sentence in tqdm(sentences, desc="Extracting relations per sentence"):
        sentence_qids = []
        sentence_doc = nlp(sentence)
        for ent in sentence_doc.ents:
            if ent._.kb_ents:
                q_id = ent._.kb_ents[0][0]
                sentence_qids.append(q_id)
        for i in range(len(sentence_qids) - 1):
            qid1 = sentence_qids[i]
            qid2 = sentence_qids[i+1]
            if knowledge_graph.has_relation(qid1, qid2):
                for rel_id in knowledge_graph.data.get(qid1, {}).keys(): #Find relation
                    if knowledge_graph.has_relation(qid1, qid2, rel_id):
                        relation_ids.add(rel_id)
                        break

        concept_ids.extend(sentence_qids) #Collect Qids in this sentence


    # --- 6. Create Integer Mappings (stoi, itos) ---
    stoi_concepts = {cid: i for i, cid in enumerate(sorted(list(knowledge_graph.concept_ids)))}
    itos_concepts = {i: cid for cid, i in stoi_concepts.items()}

    stoi_relations = {rid: i for i, rid in enumerate(sorted(list(knowledge_graph.relation_ids)))}
    itos_relations = {i: rid for rid, i in stoi_relations.items()}


    # --- 7. Encode Data ---
    encoded_concept_ids = [stoi_concepts[cid] for cid in concept_ids]

    # --- 8. Split into Train/Val ---
    n = len(encoded_concept_ids)
    train_data = encoded_concept_ids[:int(n*0.9)]
    val_data = encoded_concept_ids[int(n*0.9):]

    # --- 9. Export to Binary Files ---
    train_ids = np.array(train_data, dtype=np.uint16)  # Or uint32
    val_ids = np.array(val_data, dtype=np.uint16)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))

    # --- 10. Save Metadata ---
    meta = {
        'vocab_size': len(stoi_concepts),
        'stoi': stoi_concepts,
        'itos': itos_concepts,
        'relation_stoi': stoi_relations,
        'relation_itos': itos_relations,
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"Processed {len(concept_ids)} concept occurrences.")
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Saved metadata to {os.path.join(output_dir, 'meta.pkl')}")




if __name__ == '__main__':
    input_file = 'data/shakespeare/input.txt'
    output_dir = 'data/symbolic_shakespeare'
    tsv_path = os.path.join(output_dir, 'wikidata_subset.tsv')  # Output TSV
    wikidata_dump_path = 'wikidata/latest-truthy.nt.bz2'  # *** Path to Wikidata dump ***

    os.makedirs(output_dir, exist_ok=True)
    prepare_symbolic_data(input_file, output_dir, tsv_path, wikidata_dump_path, dataset_name="symbolic_shakespeare")
