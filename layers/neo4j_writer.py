import os
import json
import glob
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
LAYER_2C_DIR = "outputs/layer_2c"

class KnowledgeGraphWriter:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        if not uri or not user or not password:
            raise ValueError("Neo4j credentials are missing. Please check your .env file.")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("[Neo4j Writer] Successfully connected to the database.")
        except (ServiceUnavailable, AuthError) as e:
            print(f"[Neo4j Writer] ERROR: Could not connect to Neo4j. Details: {e}")
            raise

    def close(self):
        self.driver.close()

    def insert_triple(self, tx, subject: str, predicate: str, obj: str, obj_alignment: str):
        """
        Dynamically translates RDF-style semantic triples into Neo4j Property Graph format.
        """
        # 1. Handle Node Classification (rdf:type)
        if predicate == "rdf:type":
            # Example: MERGE (s:ThresholdRule {name: "RULE-ST01-01"})
            label = obj.replace(" ", "")
            query = f"MERGE (s:`{label}` {{name: $subject}})"
            tx.run(query, subject=subject)
            return

        # 2. Handle Literal Properties (Numbers, Action Text, Conditions)
        if obj_alignment == "literal":
            # Clean predicate to be a valid property name (e.g., has_warn_hi)
            prop_name = predicate.replace(":", "_")
            
            # Example: MATCH (s {name: "RULE-ST01-01"}) SET s.has_warn_hi = "26.0"
            query = f"""
            MERGE (s {{name: $subject}})
            SET s.{prop_name} = $obj
            """
            tx.run(query, subject=subject, obj=obj)
            return

        # 3. Handle Structural Relationships (Node to Node edges)
        rel_type = predicate.replace(" ", "_").upper()
        
        # Example: MERGE (s {name: "ST01_FILLING"})-[r:FEEDS_INTO]->(o {name: "ST02_SEALING"})
        query = f"""
        MERGE (s {{name: $subject}})
        MERGE (o {{name: $obj}})
        MERGE (s)-[r:`{rel_type}`]->(o)
        """
        tx.run(query, subject=subject, obj=obj)

    def process_aligned_files(self):
        """Iterates over the output of Agent 2C and pushes to Neo4j."""
        json_files = glob.glob(os.path.join(LAYER_2C_DIR, "*_agent2c_aligned.json"))
        
        if not json_files:
            print(f"No aligned files found in {LAYER_2C_DIR}. Please run Agent 2C first.")
            return

        total_triples = 0
        
        with self.driver.session(database=self.database) as session:
            for file_path in json_files:
                print(f"Processing: {os.path.basename(file_path)}")
                
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                file_triples_count = 0
                for result in data.get("results", []):
                    for relation in result.get("aligned_relations", []):
                        
                        # Correctly extract keys based on Agent 2C's output schema
                        sub = relation.get("subject")
                        pred = relation.get("predicate")
                        obj = relation.get("object")
                        obj_alignment = relation.get("_object_alignment")
                        
                        if sub and pred and obj:
                            session.execute_write(
                                self.insert_triple, 
                                sub, pred, obj, obj_alignment
                            )
                            file_triples_count += 1
                            total_triples += 1
                            
                print(f"  -> Inserted {file_triples_count} triples.")

        print(f"\n{'='*60}")
        print(f"SUCCESS: {total_triples} total triples pushed to Neo4j!")
        print(f"{'='*60}")

if __name__ == "__main__":
    print(f"{'='*60}")
    print("LAYER 4: NEO4J KNOWLEDGE GRAPH WRITER")
    print(f"{'='*60}")
    
    writer = None
    try:
        writer = KnowledgeGraphWriter()
        writer.process_aligned_files()
    except Exception as e:
        print(f"Script aborted due to error: {e}")
    finally:
        if writer:
            writer.close()
            print("Connection closed.")