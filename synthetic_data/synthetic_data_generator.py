#!/usr/bin/env python3

"""
Security RAG Generator for Synthetic Security Data

This module uses Retrieval Augmented Generation (RAG) to create realistic security events
by leveraging existing security datasets as context. It integrates with the existing
cybersecurity infrastructure and datasets.

Features:
- Loads real security data from existing datasets via SSH
- Creates vector embeddings for efficient context retrieval
- Generates realistic security events using RAG with Ollama models
- Saves generated events to MySQL and MongoDB for dashboard integration
"""

import os
import sys
import json
import time
import uuid
import random
import datetime
import argparse
import pandas as pd
import numpy as np
import pymysql
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import requests
from urllib.parse import quote_plus
import paramiko
import io

# Import LangChain components with updated imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.llms import Ollama

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("security_rag_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SecurityRAGGenerator")

# Load environment variables
load_dotenv()

# SSH connection settings
SSH_HOST = os.getenv('SSH_HOST')
SSH_USER = os.getenv('SSH_USER')
SSH_KEY_PATH = os.path.expanduser(os.getenv('SSH_KEY_PATH'))

# Dataset paths on remote server
DATASET_PATHS = {
    'intrusion_detection': os.getenv('INTRUSION_DETECTION_DATASET'),
    'ai_enhanced_events': os.getenv('AI_ENHANCED_DATASET'),
    'text_based_detection': os.getenv('TEXT_BASED_DATASET'),
    'rba_dataset': os.getenv('RBA_DATASET')
}

# Database configurations for data storage
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST'),
    'user': os.getenv('DEFAULT_MYSQL_USER'),
    'password': os.getenv('DEFAULT_MYSQL_PASSWORD'),
    'database': os.getenv('DEFAULT_MYSQL_DB'),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

MONGODB_CONFIG = {
    'uri': f"mongodb://{quote_plus(os.getenv('DEFAULT_MONGODB_USER'))}:{quote_plus(os.getenv('DEFAULT_MONGODB_PASSWORD'))}@{os.getenv('MONGODB_HOST')}:27017/{os.getenv('DEFAULT_MONGODB_DB')}?authSource={os.getenv('DEFAULT_MONGODB_DB')}",
    'database': os.getenv('DEFAULT_MONGODB_DB')
}

# Ollama API settings
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

# Create output directory for generated data and embeddings
output_dir = Path('synthetic_data')
output_dir.mkdir(exist_ok=True)

embedding_dir = output_dir / 'embeddings'
embedding_dir.mkdir(exist_ok=True)

# Simple wrapper class for direct API calls to Ollama (backup option)
class DirectOllamaAPI:
    """Simple wrapper for direct calls to Ollama API if LangChain integration fails"""
    
    def __init__(self, model_name, temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.api_url = f"{OLLAMA_API_BASE}/api/generate"
    
    def generate(self, prompt):
        """Generate text directly through the Ollama API"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": 1024,
            "stream": False
        }
        
        response = requests.post(self.api_url, json=data)
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        
        return response.json().get("response", "")


class SecurityRAGGenerator:
    """Security RAG Generator uses real security data to generate realistic security events"""
    
    def __init__(self, model_name="deepseek-r1:1.5b", embeddings_model="sentence-transformers/all-MiniLM-L6-v2", use_direct_api=False):
        """Initialize the RAG-powered synthetic data generator"""
        logger.info(f"Initializing SecurityRAGGenerator with model {model_name}")
        
        # Initialize models
        self.model_name = model_name
        self.embeddings_model_name = embeddings_model
        self.use_direct_api = use_direct_api
        
        # Check if Ollama is running and the model is available
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                if self.model_name not in models:
                    available_models = ", ".join(models)
                    logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {available_models}")
                    logger.warning(f"Will attempt to use the model anyway; Ollama may pull it automatically.")
                logger.info("Ollama server is running")
            else:
                logger.error(f"Failed to connect to Ollama API: Status code {response.status_code}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama server: {e}")
            logger.warning("Ensure Ollama is running with: ollama serve")
            raise RuntimeError("Failed to connect to Ollama")
        
        # Initialize LangChain components
        try:
            if use_direct_api:
                # Use direct API calls instead of LangChain for LLM
                self.llm = DirectOllamaAPI(model_name=model_name, temperature=0.7)
                logger.info("Using direct Ollama API")
            else:
                # Use LangChain's Ollama integration
                self.llm = Ollama(model=model_name, temperature=0.7)
                logger.info("Using LangChain Ollama integration")
                
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Initialize vector store
            self.vector_store = None
            self.event_type_stores = {}  # Separate vector stores for each event type
            
            # Track if we've loaded context
            self.context_loaded = False
            
            # SSH client for accessing datasets
            self.ssh_client = None
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.error("Please install the required packages: pip install sentence-transformers langchain-ollama")
            raise
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def connect_to_server(self):
        """Connect to the remote server via SSH."""
        try:
            logger.info(f"Connecting to {SSH_USER}@{SSH_HOST} using key {SSH_KEY_PATH}...")
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=SSH_HOST,
                username=SSH_USER,
                key_filename=SSH_KEY_PATH
            )
            logger.info("Connected to remote server")
            return client
        except Exception as e:
            logger.error(f"Error connecting to server: {e}")
            return None
    
    def read_dataset_from_server(self, dataset_name, chunksize=None, nrows=None):
        """Read a dataset from the remote server as a pandas DataFrame."""
        if self.ssh_client is None:
            self.ssh_client = self.connect_to_server()
            if self.ssh_client is None:
                logger.error("Failed to connect to SSH server")
                return None
        
        dataset_path = DATASET_PATHS.get(dataset_name)
        if not dataset_path:
            logger.error(f"Dataset path not found for {dataset_name}")
            return None
        
        try:
            # Get expanded path
            stdin, stdout, stderr = self.ssh_client.exec_command(f"echo {dataset_path}")
            expanded_path = stdout.read().decode('utf-8').strip()
            
            logger.info(f"Reading dataset from {expanded_path}")
            
            if chunksize is not None:
                # For large datasets, yield chunks
                command = f"cat {expanded_path}"
                stdin, stdout, stderr = self.ssh_client.exec_command(command)
                chunks = pd.read_csv(stdout, chunksize=chunksize)
                return chunks
            elif nrows is not None:
                # For limited rows (sample)
                command = f"head -n {nrows+1} {expanded_path}"  # +1 for header
                stdin, stdout, stderr = self.ssh_client.exec_command(command)
                df = pd.read_csv(stdout)
                return df
            else:
                # For full dataset
                command = f"cat {expanded_path}"
                stdin, stdout, stderr = self.ssh_client.exec_command(command)
                df = pd.read_csv(stdout)
                return df
        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
            return None
    
    def connect_mysql(self):
        """Connect to MySQL database"""
        try:
            connection = pymysql.connect(
                host=MYSQL_CONFIG['host'],
                user=MYSQL_CONFIG['user'],
                password=MYSQL_CONFIG['password'],
                database=MYSQL_CONFIG['database'],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            logger.info(f"Connected to MySQL: {MYSQL_CONFIG['user']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}")
            return connection
        except Exception as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return None

    def connect_mongodb(self):
        """Connect to MongoDB database"""
        try:
            client = MongoClient(MONGODB_CONFIG['uri'], serverSelectionTimeoutMS=5000)
            db = client[MONGODB_CONFIG['database']]
            # Test the connection
            client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {MONGODB_CONFIG['database']}")
            return db
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            return None
    
    def load_from_mysql(self, table_name, limit=1000):
        """Load security events from MySQL for context"""
        connection = self.connect_mysql()
        if connection is None:
            return []
        
        try:
            with connection.cursor() as cursor:
                # Check if table exists
                cursor.execute("""
                SELECT COUNT(*) as count 
                FROM information_schema.tables 
                WHERE table_schema = %s 
                AND table_name = %s
                """, (MYSQL_CONFIG['database'], table_name))
                
                result = cursor.fetchone()
                if result['count'] == 0:
                    logger.warning(f"Table {table_name} doesn't exist in MySQL")
                    return []
                
                # Query events
                query = f"""
                SELECT * FROM {table_name}
                LIMIT {limit}
                """
                cursor.execute(query)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error loading from MySQL: {e}")
            return []
        finally:
            connection.close()
    
    def load_from_mongodb(self, collection_name, limit=1000):
        """Load security events from MongoDB for context"""
        db = self.connect_mongodb()
        if db is None:
            return []
        
        try:
            # Check if collection exists
            if collection_name not in db.list_collection_names():
                logger.warning(f"Collection {collection_name} doesn't exist in MongoDB")
                return []
            
            # Query events
            collection = db[collection_name]
            events = list(collection.find().limit(limit))
            
            # Convert ObjectId to string for serialization
            for event in events:
                if '_id' in event:
                    event['_id'] = str(event['_id'])
            
            return events
        except Exception as e:
            logger.error(f"Error loading from MongoDB: {e}")
            return []
    
    def load_context_from_file(self, file_path):
        """Load context from a JSON or CSV file"""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path).to_dict('records')
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return []
        except Exception as e:
            logger.error(f"Error loading from file: {e}")
            return []
    
    def events_to_documents(self, events, event_type=None):
        """Convert security events to LangChain documents for vector storage"""
        documents = []
        
        for event in events:
            # Skip empty events
            if not event:
                continue
            
            # Convert to string if it's not already
            if isinstance(event, dict):
                event_str = json.dumps(event, indent=2)
            else:
                event_str = str(event)
            
            # Create metadata
            metadata = {
                "source": "database",
                "event_type": event_type if event_type else "unknown"
            }
            
            # If the event has specific type information, use it
            if isinstance(event, dict):
                if "event_type" in event:
                    metadata["event_type"] = event["event_type"]
                elif "_meta" in event and "event_type" in event["_meta"]:
                    metadata["event_type"] = event["_meta"]["event_type"]
            
            # Create document
            documents.append(Document(
                page_content=event_str,
                metadata=metadata
            ))
        
        return documents
    
    def split_documents(self, documents):
        """Split documents into chunks for better retrieval"""
        return self.text_splitter.split_documents(documents)
    
    def load_data_from_datasets(self):
        """Load data from actual datasets for context"""
        all_documents = []
        event_type_docs = {
            "intrusion": [],
            "authentication": [],
            "data_access": [],
            "malware": []
        }
        
        # Connect to SSH server
        self.ssh_client = self.connect_to_server()
        if self.ssh_client is None:
            logger.error("Failed to connect to SSH server for dataset access")
            return 0
        
        try:
            # Load intrusion detection dataset
            logger.info("Loading intrusion detection dataset")
            intrusion_df = self.read_dataset_from_server('intrusion_detection', nrows=5000)
            if intrusion_df is not None:
                logger.info(f"Loaded {len(intrusion_df)} rows from intrusion detection dataset")
                
                # Convert dataframe to documents
                for _, row in intrusion_df.iterrows():
                    event_dict = row.to_dict()
                    docs = self.events_to_documents([event_dict], "intrusion")
                    event_type_docs["intrusion"].extend(docs)
                    all_documents.extend(docs)
            
            # Load text-based detection dataset (if not too large)
            logger.info("Loading text-based detection dataset")
            text_df = self.read_dataset_from_server('text_based_detection', nrows=2000)
            if text_df is not None:
                logger.info(f"Loaded {len(text_df)} rows from text-based detection dataset")
                
                # Convert dataframe to documents
                for _, row in text_df.iterrows():
                    event_dict = row.to_dict()
                    docs = self.events_to_documents([event_dict], "data_access")
                    event_type_docs["data_access"].extend(docs)
                    all_documents.extend(docs)
            
            # Load a small sample from RBA dataset (it's very large)
            logger.info("Loading sample from RBA dataset")
            rba_df = self.read_dataset_from_server('rba_dataset', nrows=1000)
            if rba_df is not None:
                logger.info(f"Loaded {len(rba_df)} rows from RBA dataset")
                
                # Convert dataframe to documents
                for _, row in rba_df.iterrows():
                    event_dict = row.to_dict()
                    docs = self.events_to_documents([event_dict], "authentication")
                    event_type_docs["authentication"].extend(docs)
                    all_documents.extend(docs)
                    
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
        finally:
            # Close SSH connection
            if self.ssh_client:
                self.ssh_client.close()
                logger.info("SSH connection closed")
        
        logger.info(f"Loaded {len(all_documents)} documents from actual datasets")
        
        if not all_documents:
            logger.warning("No data loaded from datasets, falling back to database")
            return 0
            
        # Split documents into chunks
        all_chunks = self.split_documents(all_documents)
        logger.info(f"Split into {len(all_chunks)} chunks")
        
        # Create main vector store
        self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        
        # Create vector stores for each event type
        for event_type, docs in event_type_docs.items():
            if docs:
                chunks = self.split_documents(docs)
                logger.info(f"Creating vector store for {event_type} with {len(chunks)} chunks")
                self.event_type_stores[event_type] = FAISS.from_documents(chunks, self.embeddings)
        
        # Save vector stores
        self.save_vector_stores()
        
        self.context_loaded = True
        return len(all_documents)
    
    def load_data_for_context(self, mysql_tables=None, mongo_collections=None, files=None, use_datasets=True):
        """Load data from various sources to build the context for generation"""
        all_documents = []
        event_type_docs = {
            "intrusion": [],
            "authentication": [],
            "data_access": [],
            "malware": []
        }
        
        # First try to load from actual datasets if requested
        if use_datasets:
            docs_count = self.load_data_from_datasets()
            if docs_count > 0:
                logger.info(f"Successfully loaded {docs_count} documents from datasets")
                return docs_count
        
        # Load from MySQL
        if mysql_tables:
            for table in mysql_tables:
                logger.info(f"Loading data from MySQL table: {table}")
                events = self.load_from_mysql(table)
                # Try to identify event type from table name
                event_type = None
                for et in event_type_docs.keys():
                    if et in table.lower():
                        event_type = et
                        break
                
                docs = self.events_to_documents(events, event_type)
                all_documents.extend(docs)
                
                # Add to specific event type if identified
                if event_type:
                    event_type_docs[event_type].extend(docs)
        
        # Load from MongoDB
        if mongo_collections:
            for collection in mongo_collections:
                logger.info(f"Loading data from MongoDB collection: {collection}")
                events = self.load_from_mongodb(collection)
                # Try to identify event type from collection name
                event_type = None
                for et in event_type_docs.keys():
                    if et in collection.lower():
                        event_type = et
                        break
                
                docs = self.events_to_documents(events, event_type)
                all_documents.extend(docs)
                
                # Add to specific event type if identified
                if event_type:
                    event_type_docs[event_type].extend(docs)
        
        # Load from files
        if files:
            for file_path in files:
                logger.info(f"Loading data from file: {file_path}")
                events = self.load_context_from_file(file_path)
                # Try to identify event type from file name
                event_type = None
                file_name = Path(file_path).stem.lower()
                for et in event_type_docs.keys():
                    if et in file_name:
                        event_type = et
                        break
                
                docs = self.events_to_documents(events, event_type)
                all_documents.extend(docs)
                
                # Add to specific event type if identified
                if event_type:
                    event_type_docs[event_type].extend(docs)
        
        if not all_documents:
            logger.warning("No context data found.")
            return 0
        
        logger.info(f"Loaded {len(all_documents)} documents for context")
        
        # Split documents into chunks
        all_chunks = self.split_documents(all_documents)
        logger.info(f"Split into {len(all_chunks)} chunks")
        
        # Create main vector store
        self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        
        # Create vector stores for each event type
        for event_type, docs in event_type_docs.items():
            if docs:
                chunks = self.split_documents(docs)
                logger.info(f"Creating vector store for {event_type} with {len(chunks)} chunks")
                self.event_type_stores[event_type] = FAISS.from_documents(chunks, self.embeddings)
        
        # Save vector stores
        self.save_vector_stores()
        
        self.context_loaded = True
        return len(all_documents)
    
    def save_vector_stores(self):
        """Save vector stores to disk"""
        if self.vector_store:
            self.vector_store.save_local(str(embedding_dir / "main_store"))
        
        for event_type, store in self.event_type_stores.items():
            store.save_local(str(embedding_dir / f"{event_type}_store"))
    
    def load_vector_stores(self):
        """Load vector stores from disk if available"""
        main_store_path = embedding_dir / "main_store"
        if main_store_path.exists():
            try:
                self.vector_store = FAISS.load_local(str(main_store_path), self.embeddings)
                logger.info("Loaded main vector store from disk")
            except Exception as e:
                logger.error(f"Error loading main vector store: {e}")
        
        for event_type in ["intrusion", "authentication", "data_access", "malware"]:
            event_store_path = embedding_dir / f"{event_type}_store"
            if event_store_path.exists():
                try:
                    self.event_type_stores[event_type] = FAISS.load_local(str(event_store_path), self.embeddings)
                    logger.info(f"Loaded vector store for {event_type} from disk")
                except Exception as e:
                    logger.error(f"Error loading vector store for {event_type}: {e}")
        
        # Check if we loaded any vector stores
        if self.vector_store or self.event_type_stores:
            self.context_loaded = True
            return True
        return False
    
    def get_event_template(self, event_type="intrusion"):
        """Get a template for a specific type of security event"""
        templates = {
            "intrusion": {
                "timestamp": "",
                "source_ip": "",
                "destination_ip": "",
                "protocol": "",
                "port": 0,
                "attack_type": "",
                "severity": "",
                "description": "",
                "indicators_of_compromise": []
            },
            "authentication": {
                "timestamp": "",
                "username": "",
                "source_ip": "",
                "device_type": "",
                "browser": "",
                "success": True,
                "failure_reason": "",
                "location": "",
                "severity": "",
                "description": ""
            },
            "data_access": {
                "timestamp": "",
                "username": "",
                "source_ip": "",
                "data_resource": "",
                "access_type": "",
                "bytes_transferred": 0,
                "duration": 0,
                "severity": "",
                "description": "",
                "indicators": []
            },
            "malware": {
                "timestamp": "",
                "hostname": "",
                "malware_name": "",
                "file_path": "",
                "file_hash": "",
                "severity": "",
                "tactics": [],
                "techniques": [],
                "description": "",
                "indicators_of_compromise": []
            }
        }
        
        return templates.get(event_type, templates["intrusion"])
    
    def get_retrieval_prompt(self, event_type="intrusion", is_malicious=True):
        """Create a prompt template with examples from context for better retrieval"""
        # Base prompts for different event types
        base_templates = {
            "intrusion": """
            You are a security event generator that creates realistic network intrusion events.
            
            Use the following examples as reference for how to structure and format your response:
            
            {context}
            
            Now, generate a new, unique {malicious_type} network intrusion event in JSON format.
            Include the following fields: timestamp, source_ip, destination_ip, protocol, 
            port, attack_type, severity, description, indicators_of_compromise.
            
            Ensure the event details are realistic, coherent, and internally consistent.
            ONLY output valid JSON without any additional explanations or text.
            """,
            
            "authentication": """
            You are a security event generator that creates realistic authentication events.
            
            Use the following examples as reference for how to structure and format your response:
            
            {context}
            
            Now, generate a new, unique {malicious_type} authentication event in JSON format.
            Include the following fields: timestamp, username, source_ip, device_type, 
            browser, success, failure_reason, location, severity, description.
            
            Ensure the event details are realistic, coherent, and internally consistent.
            ONLY output valid JSON without any additional explanations or text.
            """,
            
            "data_access": """
            You are a security event generator that creates realistic data access events.
            
            Use the following examples as reference for how to structure and format your response:
            
            {context}
            
            Now, generate a new, unique {malicious_type} data access event in JSON format.
            Include the following fields: timestamp, username, source_ip, data_resource, 
            access_type, bytes_transferred, duration, severity, description, indicators.
            
            Ensure the event details are realistic, coherent, and internally consistent.
            ONLY output valid JSON without any additional explanations or text.
            """,
            
            "malware": """
            You are a security event generator that creates realistic malware detection events.
            
            Use the following examples as reference for how to structure and format your response:
            
            {context}
            
            Now, generate a new, unique {malicious_type} malware detection event in JSON format.
            Include the following fields: timestamp, hostname, malware_name, file_path, 
            file_hash, severity, tactics, techniques, description, indicators_of_compromise.
            
            Ensure the event details are realistic, coherent, and internally consistent.
            ONLY output valid JSON without any additional explanations or text.
            """
        }
        
        # Get the base template for the specified event type, or use intrusion as default
        template = base_templates.get(event_type, base_templates["intrusion"])
        
        # Set malicious type
        malicious_type = "malicious" if is_malicious else "benign"
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "malicious_type"]
        )
    
    def generate_event(self, event_type="intrusion", is_malicious=True, temperature=0.7):
        """Generate a synthetic security event using RAG"""
        logger.info(f"Generating {event_type} event (malicious={is_malicious})")
        
        # Check if context is loaded, if not try to load from disk or use defaults
        if not self.context_loaded:
            loaded = self.load_vector_stores()
            if not loaded:
                logger.warning("No context loaded and no saved vector stores found. Loading from datasets.")
                # Try to load from datasets
                self.load_data_from_datasets()
        
        # Get the appropriate vector store
        retriever = None
        if event_type in self.event_type_stores:
            # Use the event-specific store
            retriever = self.event_type_stores[event_type].as_retriever(
                search_kwargs={"k": 5}
            )
            logger.info(f"Using event-specific vector store for {event_type}")
        elif self.vector_store:
            # Fall back to the main store
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            )
            logger.info("Using main vector store")
        else:
            logger.warning("No vector stores available")
            # We'll proceed with just the prompt template
        
        # Create the prompt
        prompt = self.get_retrieval_prompt(event_type, is_malicious)
        
        # Set malicious type string
        malicious_type = "malicious" if is_malicious else "benign"
        
        # Try to generate with retrieval if we have a retriever
        if retriever and not self.use_direct_api:
            # Create a retrieval chain
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    self.llm,
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
                
                # Run the chain
                result = qa_chain({"query": f"Generate a {malicious_type} {event_type} event"})
                generated_text = result["result"]
                
            except Exception as e:
                logger.error(f"Error in retrieval chain: {e}")
                logger.info("Falling back to direct generation")
                
                # Fall back to direct generation
                if self.use_direct_api:
                    # Use direct API call
                    context = "No examples available due to retrieval error."
                    full_prompt = prompt.format(context=context, malicious_type=malicious_type)
                    generated_text = self.llm.generate(full_prompt)
                else:
                    # Use LangChain without retrieval
                    try:
                        chain = LLMChain(llm=self.llm, prompt=prompt)
                        result = chain.run(context="No examples available due to retrieval error.", malicious_type=malicious_type)
                        generated_text = result
                    except Exception as e2:
                        logger.error(f"Error in fallback generation: {e2}")
                        # Last resort: direct API call
                        logger.info("Falling back to direct API call")
                        direct_api = DirectOllamaAPI(model_name=self.model_name)
                        full_prompt = prompt.format(context="No examples available.", malicious_type=malicious_type)
                        generated_text = direct_api.generate(full_prompt)
        else:
            # Direct generation without retrieval
            if self.use_direct_api:
                # Use direct API call
                context = "No examples available."
                full_prompt = prompt.format(context=context, malicious_type=malicious_type)
                generated_text = self.llm.generate(full_prompt)
            else:
                # Use LangChain without retrieval
                try:
                    chain = LLMChain(llm=self.llm, prompt=prompt)
                    result = chain.run(context="No examples available.", malicious_type=malicious_type)
                    generated_text = result
                except Exception as e:
                    logger.error(f"Error in direct generation: {e}")
                    # Last resort: direct API call
                    logger.info("Falling back to direct API call")
                    direct_api = DirectOllamaAPI(model_name=self.model_name)
                    full_prompt = prompt.format(context="No examples available.", malicious_type=malicious_type)
                    generated_text = direct_api.generate(full_prompt)
        
        # Try to extract valid JSON from the response
        try:
            # Find the first '{' and the last '}'
            start_idx = generated_text.find('{')
            end_idx = generated_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = generated_text[start_idx:end_idx+1].strip()
                
                # Parse as JSON to validate and normalize
                generated_data = json.loads(json_str)
                
                # Add metadata
                generated_data["_meta"] = {
                    "generated_at": datetime.datetime.now().isoformat(),
                    "event_type": event_type,
                    "is_malicious": is_malicious,
                    "model": self.model_name,
                    "parameters": {
                        "temperature": temperature
                    }
                }
                
                # Log success
                logger.info(f"Successfully generated {event_type} event")
                
                return generated_data
            else:
                logger.error("Failed to extract JSON from model output")
                logger.debug(f"Generated text: {generated_text}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse generated text as JSON: {e}")
            logger.debug(f"Generated text: {generated_text}")
            return None
    
    def generate_batch(self, batch_size=10, event_types=None, malicious_ratio=0.3):
        """Generate a batch of security events with varying types and severities"""
        
        if event_types is None:
            event_types = ["intrusion", "authentication", "data_access", "malware"]
        
        generated_events = []
        
        for i in range(batch_size):
            # Randomly select event type and malicious status
            event_type = random.choice(event_types)
            is_malicious = random.random() < malicious_ratio
            
            # Add some randomness to generation parameters
            temperature = random.uniform(0.6, 0.9)
            
            logger.info(f"Generating event {i+1}/{batch_size} - Type: {event_type}, Malicious: {is_malicious}")
            
            # Generate the data
            event_data = self.generate_event(event_type, is_malicious, temperature)
            
            if event_data:
                generated_events.append(event_data)
            else:
                logger.warning(f"Failed to generate event {i+1}/{batch_size}")
        
        return generated_events
    
    def save_to_file(self, events, filename=None):
        """Save generated events to a file"""
        if not events:
            logger.warning("No events to save")
            return
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"rag_events_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(events, f, indent=2)
            logger.info(f"Saved {len(events)} events to {filename}")
        except Exception as e:
            logger.error(f"Error saving events to file: {e}")
    
    def save_to_mysql(self, events, connection=None):
        """Save generated events to MySQL database"""
        if not events:
            logger.warning("No events to save to MySQL")
            return
        
        # Connect to MySQL if connection not provided
        close_connection = False
        if connection is None:
            connection = self.connect_mysql()
            close_connection = True
        
        if connection is None:
            logger.error("Failed to connect to MySQL")
            return
        
        try:
            with connection.cursor() as cursor:
                # Check if table exists, create if not
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_security_events (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    event_data JSON,
                    event_type VARCHAR(50),
                    is_malicious TINYINT(1),
                    severity VARCHAR(20),
                    timestamp DATETIME,
                    generated_at DATETIME,
                    INDEX (event_type),
                    INDEX (is_malicious),
                    INDEX (severity),
                    INDEX (timestamp)
                )
                """)
                connection.commit()
                
                # Insert events
                for event in events:
                    # Extract metadata
                    meta = event.get("_meta", {})
                    event_type = meta.get("event_type", "unknown")
                    is_malicious = 1 if meta.get("is_malicious", False) else 0
                    
                    # Extract common fields
                    severity = event.get("severity", "unknown")
                    timestamp_str = event.get("timestamp", 
                                           datetime.datetime.now().isoformat())
                    
                    # Try to parse timestamp
                    try:
                        if 'T' in timestamp_str:
                            timestamp = datetime.datetime.fromisoformat(timestamp_str)
                        else:
                            timestamp = datetime.datetime.strptime(timestamp_str, 
                                                                "%Y-%m-%d %H:%M:%S")
                    except:
                        timestamp = datetime.datetime.now()
                    
                    # Current time for generated_at
                    generated_at = datetime.datetime.now()
                    
                    # Insert into database
                    cursor.execute("""
                    INSERT INTO rag_security_events 
                    (event_data, event_type, is_malicious, severity, timestamp, generated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        json.dumps(event),
                        event_type,
                        is_malicious,
                        severity,
                        timestamp,
                        generated_at
                    ))
                
                connection.commit()
                logger.info(f"Saved {len(events)} events to MySQL")
                
        except Exception as e:
            logger.error(f"Error saving to MySQL: {e}")
        finally:
            if close_connection and connection:
                connection.close()
    
    def save_to_mongodb(self, events, db=None):
        """Save generated events to MongoDB"""
        if not events:
            logger.warning("No events to save to MongoDB")
            return
        
        # Connect to MongoDB if connection not provided
        close_connection = False
        if db is None:
            db = self.connect_mongodb()
            close_connection = True
        
        if db is None:
            logger.error("Failed to connect to MongoDB")
            return
        
        try:
            # Get the collection
            collection = db["rag_security_events"]
            
            # Insert events
            result = collection.insert_many(events)
            
            logger.info(f"Saved {len(result.inserted_ids)} events to MongoDB")
            
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")

def main():
    """Main function to run the RAG-powered generator"""
    parser = argparse.ArgumentParser(description="Generate synthetic security data using RAG")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="deepseek-r1:1.5b", 
                        help="Ollama model to use for generation")
    parser.add_argument("--embeddings", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embeddings model to use for vector store")
    parser.add_argument("--direct-api", action="store_true",
                       help="Use direct Ollama API calls instead of LangChain")
    
    # Dataset options
    parser.add_argument("--no-datasets", action="store_true",
                       help="Skip loading from original datasets")
    
    # Context loading options
    parser.add_argument("--mysql-tables", type=str, 
                       help="Comma-separated list of MySQL tables to load for context")
    parser.add_argument("--mongo-collections", type=str, 
                       help="Comma-separated list of MongoDB collections to load for context")
    parser.add_argument("--context-files", type=str, 
                       help="Comma-separated list of files to load for context")
    parser.add_argument("--skip-context-loading", action="store_true",
                       help="Skip loading context and use existing vector stores if available")
    
    # Generation parameters
    parser.add_argument("--num-events", type=int, default=10, 
                       help="Number of events to generate")
    parser.add_argument("--event-types", type=str, 
                       help="Comma-separated list of event types to generate")
    parser.add_argument("--malicious-ratio", type=float, default=0.3, 
                       help="Ratio of malicious to benign events (0.0-1.0)")
    
    # Output options
    parser.add_argument("--output-file", type=str, 
                       help="Path to save generated events as JSON")
    parser.add_argument("--mysql", action="store_true", 
                       help="Save events to MySQL database")
    parser.add_argument("--mongodb", action="store_true", 
                       help="Save events to MongoDB database")
    parser.add_argument("--output-all", action="store_true", 
                       help="Save events to file, MySQL, and MongoDB")
    
    args = parser.parse_args()
    
    logger.info("Starting RAG-powered synthetic security data generator")
    
    # Initialize generator
    try:
        generator = SecurityRAGGenerator(
            model_name=args.model, 
            embeddings_model=args.embeddings,
            use_direct_api=args.direct_api
        )
        logger.info("Generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return
    
    # Load context if needed
    if not args.skip_context_loading:
        mysql_tables = args.mysql_tables.split(',') if args.mysql_tables else []
        mongo_collections = args.mongo_collections.split(',') if args.mongo_collections else []
        context_files = args.context_files.split(',') if args.context_files else []
        
        # If no sources specified but not skipping, use defaults
        if not mysql_tables and not mongo_collections and not context_files:
            mysql_tables = ["synthetic_security_events", "rag_security_events"]
            mongo_collections = ["synthetic_security_events", "rag_security_events"]
            # Look for any JSON files in the output directory
            json_files = list(output_dir.glob("*.json"))
            context_files = [str(f) for f in json_files]
        
        num_docs = generator.load_data_for_context(
            mysql_tables=mysql_tables,
            mongo_collections=mongo_collections,
            files=context_files,
            use_datasets=not args.no_datasets
        )
        logger.info(f"Loaded {num_docs} documents for context")
    
    # Generate events
    event_types = args.event_types.split(',') if args.event_types else None
    events = generator.generate_batch(
        batch_size=args.num_events,
        event_types=event_types,
        malicious_ratio=args.malicious_ratio
    )
    
    # Save the generated events
    if args.output_file or args.output_all:
        output_file = args.output_file if args.output_file else None
        generator.save_to_file(events, output_file)
    
    if args.mysql or args.output_all:
        generator.save_to_mysql(events)
    
    if args.mongodb or args.output_all:
        generator.save_to_mongodb(events)
    
    logger.info(f"Generated {len(events)} synthetic security events using RAG")

if __name__ == "__main__":
    main()