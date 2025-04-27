# TODO: Implement RBA dataset loading (Shreyaa)
from datetime import datetime
from database.database_loader import DATASET_PATHS, read_dataset_from_server
from tqdm import tqdm
import pandas as pd

def load_rba_dataset_to_database(ssh_client,mysql_conn):
    try :
        if ssh_client is None or mysql_conn is None :
            return False
        
        #Create Table
        with mysql_conn.cursor() as cursor :
            cursor.execute("""
            DROP TABLE IF EXISTS rba_logins;
            """)

            cursor.execute("""
            CREATE TABLE rba_logins (
            id INT AUTO_INCREMENT PRIMARY KEY,
            login_timestamp DATETIME(3),
            userid VARCHAR(25),
            round_trip_time_ms FLOAT,
            ip_address VARCHAR(45),
            country VARCHAR(5),
            region VARCHAR(100),
            city VARCHAR(100),
            asn VARCHAR(100),
            user_agent TEXT
            );
            """)
            mysql_conn.commit()


        #Read dataset from remote server
        chunks = read_dataset_from_server(ssh_client,DATASET_PATHS['rba_dataset'],chunksize=10000)
        if chunks is None :
            print(f"❌ Dataset could not be read")
            return False

        #Security state
        slow_login_count = {}
        ip_user_map = {}
        user_geo = {}
        
        total_inserted=0
        approx = 33000000 // 10000
        chunk_progress = tqdm(chunks, total = approx,desc="Loading Chunks", unit="chunk")


        for chunk in chunk_progress :
            rows_to_insert = []

            for _, row in chunk.iterrows() :
                try:
                    #Handle timestamp
                    ts = row['Login Timestamp']
                    if isinstance(ts,str):
                        login_timestamp = pd.to_datetime(row['Login Timestamp'], errors='coerce')
                    else:
                        login_timestamp=None

                    #Handle userid
                    user_id = str(row['User ID'])
                    
                    #Prepare data row
                    rows_to_insert.append((
                        login_timestamp,
                        user_id,
                        row['Round-Trip Time [ms]'] if not pd.isnull(row['Round-Trip Time [ms]']) else None,
                        row['IP Address']if not pd.isnull(row['IP Address']) else None,
                        row['Country'] if not pd.isnull(row['Country']) else None,
                        row['Region'] if not pd.isnull(row['Region']) else None,
                        row['City'] if not pd.isnull(row['City']) else None,
                        row['ASN'] if not pd.isnull(row['ASN']) else None,
                        row['User Agent String'] if not pd.isnull(row['User Agent String']) else None
                    ))
                except Exception as row_err :
                    print(f"⚠️ Skipping row due to error: {row_err}")
                
            #Insert chunk
            if rows_to_insert:
                with mysql_conn.cursor() as cursor :
                    cursor.executemany("""
                    INSERT INTO rba_logins(
                    login_timestamp, userid, round_trip_time_ms,
                    ip_address, country, region, city,
                    asn, user_agent) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, rows_to_insert)
                    mysql_conn.commit()
                
                total_inserted += len(rows_to_insert)
                chunk_progress.set_postfix(inserted=total_inserted)
                

        print("🎉 Dataset successfully loaded into MySQL.")
        return True

    except Exception as e:
        print(f"❌ Error in load_rba_dataset_to_database: {e}")
        return False
