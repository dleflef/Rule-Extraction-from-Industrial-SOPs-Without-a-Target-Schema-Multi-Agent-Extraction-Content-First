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
            # Verify connectivity
            self.driver.verify_connectivity()
            print("[Neo4j Writer] Successfully connected to the database.")
        except (ServiceUnavailable, AuthError) as e:
            print(f"[Neo4j Writer] ERROR: Could not connect to Neo4j. Details: {e}")
            raise

    def close(self):
        self.driver.close()

    def insert_triple(self, tx, subject: str, subject_type: str, predicate: str, obj: str, obj_type: str):
        """
        Dynamically generates and runs a Cypher MERGE statement.
        MERGE ensures we don't create duplicate nodes or relationships.
        """
        # Clean labels to ensure they are valid Cypher syntax (no spaces)
        subj_label = subject_type.replace(" ", "") if subject_type else "Entity"
        obj_label = obj_type.replace(" ", "") if obj_type else "Entity"
        
        # Cypher relationships are conventionally UPPERCASE_WITH_UNDERSCORES
        rel_type = predicate.replace(" ", "_").upper()

        # Cypher does not allow parameterizing Node Labels or Relationship Types natively.
        # We must use string formatting for the labels, but we PARAMETERIZE the data (name) to prevent injection.
        query = f"""
        MERGE (s:`{subj_label}` {{name: $subject}})
        MERGE (o:`{obj_label}` {{name: $obj}})
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
        
        # Open a session to Neo4j
        with self.driver.session(database=self.database) as session:
            for file_path in json_files:
                print(f"Processing: {os.path.basename(file_path)}")
                
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                file_triples_count = 0
                for result in data.get("results", []):
                    for relation in result.get("aligned_relations", []):
                        # Extract the data
                        sub = relation.get("subject")
                        sub_type = relation.get("subject_type")
                        pred = relation.get("predicate")
                        obj = relation.get("object")
                        obj_type = relation.get("object_type")
                        
                        # Only insert if the triple is complete
                        if sub and pred and obj:
                            # Execute the transaction
                            session.execute_write(
                                self.insert_triple, 
                                sub, sub_type, pred, obj, obj_type
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