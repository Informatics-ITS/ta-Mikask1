from pinecone import Pinecone
import pandas as pd
import ast

pc = Pinecone(api_key="pcsk_5QDQPf_Q7jQsWbeArLGK9Ksk5QvdQ7cuyyVYHiJxsuaZYAZ7yJG7yxX7VYb6LXViDWQjaJ")
index = pc.Index("uu-index")

deleted_records = pd.read_csv("DELETED.csv")

def delete_record(records):
    for _, record in records.iterrows():
        name = record['name']
        
        filter = {
            "nama": {"$eq": name}
        }

        query_response = index.query(
            vector=[0]*1024,
            filter=filter,
            top_k=500,
            include_metadata=True
        )
        
        if len(query_response['matches']) > 0:
            for match in query_response['matches']:
                if match['metadata']['nama'] == name:
                    index.delete(ids=[match['id']])

if __name__ == "__main__":
    delete_record(deleted_records)


