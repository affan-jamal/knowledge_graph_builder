"""
Knowledge-Graph Builder using Hugging Face SentenceTransformer + Neo4j
No API keys required (local embeddings)
"""

import os
import sys
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ---------- Config ---------- #
load_dotenv()  # Load .env for Neo4j config

# Neo4j connection
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

# Local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim vectors

THRESH = 0.80    # similarity threshold
TOP_K = 3        # top results in similarity search
# ---------------------------- #

def embed(text: str) -> list[float]:
    """
    Generate local embedding using SentenceTransformer.
    """
    vec = model.encode(text)
    return vec.tolist()

def store_node(tx, text, vec):
    tx.run(
        "CREATE (d:Document {text:$t, embedding:$v})",
        t=text,
        v=vec
    )

def create_similarity_edges(tx, id1, vec1, id2, vec2):
    score = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    if score >= THRESH:
        tx.run(
            """
            MATCH (a:Document), (b:Document)
            WHERE id(a) = $id1 AND id(b) = $id2
            MERGE (a)-[:SIMILAR {score:$s}]->(b)
            """,
            id1=id1,
            id2=id2,
            s=score
        )

def fetch_nodes(tx):
    result = tx.run("MATCH (d:Document) RETURN id(d) AS id, d.embedding AS v")
    return {record["id"]: record["v"] for record in result}

def cosine_query(tx, qvec, k=TOP_K):
    result = tx.run(
        """
        MATCH (d:Document)
        WITH d,
             vector.similarity.cosine(d.embedding, $vec) AS score
        RETURN d.text AS txt, score
        ORDER BY score DESC
        LIMIT $k
        """,
        vec=qvec,
        k=k
    )
    return result.data()

def reset_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")

if __name__ == "__main__":
    seed_texts = [
        "Python is a versatile programming language.",
        "Neo4j excels at storing and traversing relationships.",
        "Embeddings capture the semantics of a sentence.",
    ]

    query_text = "Which database treats relationships as first-class data?"

    try:
        with driver.session() as sess:
            sess.execute_write(reset_graph)

            for text in seed_texts:
                vec = embed(text)
                sess.execute_write(store_node, text, vec)

            id_to_vec = sess.execute_read(fetch_nodes)

            from itertools import combinations
            for (id1, vec1), (id2, vec2) in combinations(id_to_vec.items(), 2):
                sess.execute_write(create_similarity_edges, id1, vec1, id2, vec2)

            qvec = embed(query_text)
            results = sess.execute_read(cosine_query, qvec)

        print("\nStored nodes:")
        for t in seed_texts:
            print(" •", t)

        print("\nQuery ➜", query_text)
        for r in results:
            print(f"  {r['score']:.3f} → {r['txt']}")

    except Exception as e:
        sys.exit(f"[ERROR] {e}")
    finally:
        driver.close()
