import streamlit as st
from dotenv import load_dotenv
import os
import sqlite3
import google.generativeai as genai
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from pymongo import MongoClient
from bson import json_util, ObjectId
import tempfile
import pandas as pd
from io import BytesIO
import json
import re
import pymysql
from urllib.parse import quote_plus
import time

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize MongoDB client at module level
mongo_client = None

def initialize_mongodb():
    """Initialize MongoDB connection with proper error handling"""
    global mongo_client
    try:
        # First try local connection
        client = MongoClient(
            "mongodb://localhost:27017/",
            serverSelectionTimeoutMS=5000,
            socketTimeoutMS=30000
        )
        client.admin.command('ping')
        mongo_client = client
        st.sidebar.success("✅ Connected to local MongoDB!")
        return client
    except Exception as local_error:
        st.sidebar.warning(f"⚠ Local MongoDB connection failed: {str(local_error)}")
        
        # Try Atlas connection if available
        atlas_uri = os.getenv("MONGODB_ATLAS_URI")
        if atlas_uri:
            try:
                client = MongoClient(
                    atlas_uri,
                    serverSelectionTimeoutMS=5000,
                    socketTimeoutMS=30000
                )
                client.admin.command('ping')
                mongo_client = client
                st.sidebar.success("✅ Connected to MongoDB Atlas!")
                return client
            except Exception as atlas_error:
                st.sidebar.error(f"❌ MongoDB Atlas connection failed: {str(atlas_error)}")
        return None

def get_connection_details(db_type):
    """Get database connection details from user with validation"""
    st.sidebar.subheader(f"{db_type.upper()} Connection Details")
    details = {}
    
    if db_type == "mysql":
        details['host'] = st.sidebar.text_input("MySQL Host", value="localhost")
        details['port'] = st.sidebar.number_input("MySQL Port", value=3306, min_value=1, max_value=65535)
        details['user'] = st.sidebar.text_input("MySQL Username", value="root")
        details['password'] = st.sidebar.text_input("MySQL Password", type="password")
        details['database'] = st.sidebar.text_input("MySQL Database Name")
        
        if st.sidebar.button("Test MySQL Connection"):
            try:
                conn = pymysql.connect(
                    host=details['host'],
                    port=int(details['port']),
                    user=details['user'],
                    password=details['password'],
                    connect_timeout=5
                )
                with conn.cursor() as cursor:
                    cursor.execute("SELECT VERSION()")
                    version = cursor.fetchone()
                    st.sidebar.success(f"✅ Connected to MySQL {version[0]}!")
                conn.close()
            except pymysql.Error as e:
                st.sidebar.error(f"❌ Connection failed: {str(e)}")
                st.sidebar.info("Troubleshooting:")
                st.sidebar.info("1. Ensure MySQL server is running")
                st.sidebar.info("2. Verify username/password")
                st.sidebar.info("3. Check host/port configuration")

    elif db_type == "postgresql":
        details['host'] = st.sidebar.text_input("PostgreSQL Host", value="localhost")
        details['port'] = st.sidebar.number_input("PostgreSQL Port", value=5432, min_value=1, max_value=65535)
        details['user'] = st.sidebar.text_input("PostgreSQL Username", value="postgres")
        details['password'] = st.sidebar.text_input("PostgreSQL Password", type="password")
        details['database'] = st.sidebar.text_input("PostgreSQL Database Name")
        
        if st.sidebar.button("Test PostgreSQL Connection"):
            try:
                engine = create_engine(
                    f"postgresql+psycopg2://{details['user']}:{details['password']}@"
                    f"{details['host']}:{details['port']}/postgres",
                    connect_args={'connect_timeout': 5}
                )
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                st.sidebar.success("✅ Connection successful!")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {str(e)}")

    elif db_type == "sqlite":
        uploaded_file = st.sidebar.file_uploader("Upload SQLite Database", type=["db", "sqlite"])
        if uploaded_file:
            temp_dir = tempfile.mkdtemp()
            db_path = os.path.join(temp_dir, "temp.db")
            with open(db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            details['database'] = db_path
            st.sidebar.success("Database uploaded successfully!")

    elif db_type == "mongodb":
        global mongo_client
        
        # Initialize MongoDB client if not already connected
        if mongo_client is None:
            mongo_client = initialize_mongodb()
        
        # Section 1: File Upload Option
        st.sidebar.markdown("**Option 1: Upload JSON/BSON File**")
        uploaded_file = st.sidebar.file_uploader(
            "Choose MongoDB data file", 
            type=["json", "bson"],
            accept_multiple_files=False,
            key="mongodb_upload"
        )
        
        if uploaded_file:
            try:
                if mongo_client is None:
                    raise Exception("No MongoDB connection available. Please connect to MongoDB first.")
                
                # Create temporary database with timestamp
                db_name = f"temp_db_{int(time.time())}"
                db = mongo_client[db_name]
                
                # Process uploaded file content
                content = uploaded_file.getvalue()
                
                # Try parsing as JSON first
                try:
                    content_str = content.decode('utf-8')
                    try:
                        # Case 1: JSON array
                        data = json.loads(content_str)
                        if isinstance(data, list):
                            db["imported_data"].insert_many(data)
                        else:
                            db["imported_data"].insert_one(data)
                    except json.JSONDecodeError:
                        # Case 2: Line-delimited JSON
                        docs = []
                        for line in content_str.splitlines():
                            line = line.strip()
                            if line:
                                try:
                                    docs.append(json.loads(line))
                                except json.JSONDecodeError:
                                    st.sidebar.warning(f"Skipped invalid JSON line: {line[:50]}...")
                        if docs:
                            db["imported_data"].insert_many(docs)
                            
                except UnicodeDecodeError:
                    # Case 3: BSON file
                    try:
                        from bson import decode_all
                        db["imported_data"].insert_many(decode_all(content))
                    except Exception as e:
                        st.sidebar.error(f"BSON parse error: {str(e)}")
                        return details
                
                details['database'] = db_name
                st.sidebar.success(f"✅ Imported {uploaded_file.name} successfully!")
                st.sidebar.info(f"Temporary database: {db_name}")
                
            except Exception as e:
                st.sidebar.error(f"❌ Import failed: {str(e)}")
                return details
        
        # Section 2: Connection Option
        st.sidebar.markdown("**Option 2: Connect to Existing MongoDB**")
        
        if mongo_client is None:
            # Atlas connection fallback
            atlas_uri = st.sidebar.text_input(
                "MongoDB Atlas URI",
                help="Format: mongodb+srv://<username>:<password>@cluster.mongodb.net/"
            )
            if st.sidebar.button("Connect to MongoDB Atlas"):
                try:
                    client = MongoClient(
                        atlas_uri,
                        serverSelectionTimeoutMS=5000,
                        socketTimeoutMS=30000
                    )
                    client.admin.command('ping')
                    mongo_client = client
                    st.sidebar.success("✅ MongoDB Atlas connection successful!")
                except Exception as e:
                    st.sidebar.error(f"❌ Atlas connection failed: {str(e)}")
        
        if mongo_client and not uploaded_file:
            try:
                db_names = mongo_client.list_database_names()
                details['database'] = st.sidebar.selectbox(
                    "Select Database", 
                    db_names,
                    help="System databases are hidden"
                )
            except Exception as e:
                st.sidebar.error(f"Failed to list databases: {str(e)}")
    
    return details

def get_schema_info(db_type, connection_details):
    """Retrieve database schema with comprehensive error handling"""
    try:
        schema_info = {}
        
        if db_type == "mysql":
            try:
                # URL encode password for special characters
                password = quote_plus(connection_details['password'])
                
                engine = create_engine(
                    f"mysql+pymysql://{connection_details['user']}:{password}@"
                    f"{connection_details['host']}:{connection_details['port']}/"
                    f"{connection_details['database']}",
                    pool_pre_ping=True,
                    connect_args={
                        'connect_timeout': 10,
                        'ssl': {'ssl_disabled': True}
                    }
                )
                
                # Test connection first
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                # Get schema if connection works
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                
                for table in tables:
                    columns = inspector.get_columns(table)
                    schema_info[table] = {
                        'fields': [col['name'] for col in columns],
                        'types': [str(col['type']) for col in columns],
                        'sample': get_table_sample(engine, table)
                    }
                
                return schema_info
            
            except OperationalError as e:
                raise Exception(
                    f"MySQL connection failed. Please check:\n"
                    f"1. MySQL server is running on {connection_details['host']}:{connection_details['port']}\n"
                    f"2. User '{connection_details['user']}' has access\n"
                    f"3. Password is correct\n"
                    f"4. No firewall blocking the port\n\n"
                    f"Technical error: {str(e)}"
                )
        
        elif db_type == "postgresql":
            try:
                engine = create_engine(
                    f"postgresql+psycopg2://{connection_details['user']}:{connection_details['password']}@"
                    f"{connection_details['host']}:{connection_details['port']}/"
                    f"{connection_details['database']}"
                )
                
                with engine.connect() as conn:
                    inspector = inspect(conn)
                    tables = inspector.get_table_names()
                    
                    for table in tables:
                        columns = inspector.get_columns(table)
                        schema_info[table] = {
                            'fields': [col['name'] for col in columns],
                            'types': [str(col['type']) for col in columns],
                            'sample': get_table_sample(engine, table)
                        }
                
                return schema_info
            
            except Exception as e:
                raise Exception(f"PostgreSQL schema retrieval failed: {str(e)}")
        
        elif db_type == "sqlite":
            conn = None
            try:
                conn = sqlite3.connect(connection_details['database'])
                cursor = conn.cursor()
                
                # Get tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    schema_info[table_name] = {
                        'fields': [col[1] for col in columns],
                        'types': [col[2] for col in columns],
                        'sample': get_sqlite_sample(conn, table_name)
                    }
                
                return schema_info
            
            except Exception as e:
                raise Exception(f"SQLite schema retrieval failed: {str(e)}")
            finally:
                if conn:
                    conn.close()
        
        elif db_type == "mongodb":
            if not mongo_client:
                raise Exception("MongoDB connection not available")
            
            try:
                db = mongo_client[connection_details['database']]
                for collection in db.list_collection_names():
                    sample = db[collection].find_one({}, {'_id': 0}) or {}
                    schema_info[collection] = {
                        'fields': list(sample.keys()),
                        'sample': sample
                    }
                return schema_info
            
            except Exception as e:
                raise Exception(f"MongoDB schema retrieval failed: {str(e)}")
        
        return {"error": "No schema information found"}
    
    except Exception as e:
        return {"error": f"Schema retrieval failed: {str(e)}"}

def get_table_sample(engine, table_name, limit=3):
    """Get sample rows from SQL table"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
            columns = result.keys()
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]
    except:
        return []

def get_sqlite_sample(conn, table_name, limit=3):
    """Get sample rows from SQLite table"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    except:
        return []

def get_gemini_response(question, schema_info, db_type, sample_docs=None):
    """Generate database queries with strict validation"""
    try:
        if isinstance(schema_info, dict) and "error" in schema_info:
            raise ValueError(schema_info["error"])
        
        if db_type == "mongodb":
            sample_info = ""
            if any('sample' in v for v in schema_info.values()):
                sample_info = "\nSample Documents:\n" + "\n".join(
                    f"{col}: {json.dumps(val['sample'], indent=2)}" 
                    for col, val in schema_info.items() if 'sample' in val
                )

            prompt = f"""
            Generate a MongoDB query for: "{question}"

            STRICT REQUIREMENTS:
            1. Use format: db.collection.command({{"key": "value"}})
            2. All property names MUST use double quotes
            3. Collections available: {list(schema_info.keys())}
            4. Return ONLY the executable query (no explanations)
            5. For find operations, always include _id: 0 unless needed

            Database Schema:
            {json.dumps({k: v['fields'] for k, v in schema_info.items()}, indent=2)}
            {sample_info}

            MongoDB Query:
            """

        else:  # SQL databases
            prompt = f"""
            Generate {db_type.upper()} SQL for: "{question}"

            STRICT REQUIREMENTS:
            1. Use proper {db_type.upper()} syntax
            2. Single line only
            3. No code blocks or backticks
            4. No explanatory text
            5. Include necessary JOINs for relations

            Database Schema:
            {json.dumps({k: {'fields': v['fields'], 'types': v['types']} 
                        for k, v in schema_info.items()}, indent=2)}

            {db_type.upper()} Query:
            """

        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        raw_query = response.text.strip()

        # Clean and validate the query
        query = re.sub(r'^(`+|javascript|sql|mongodb|\n)', '', raw_query, flags=re.IGNORECASE)
        query = re.sub(r'`+$', '', query)
        query = ' '.join(query.split()).strip()

        # MongoDB-specific validation
        if db_type == "mongodb":
            if not query.startswith("db."):
                query = f"db.{query}" if not query.startswith("db.") else query
            if not re.match(r'^db\.\w+\.\w+\(.*\)$', query):
                raise ValueError(f"Invalid MongoDB query format: {query[:200]}...")
            if not any(f"db.{col}." in query for col in schema_info.keys()):
                raise ValueError(f"Collection not found in schema: {query}")

        return query

    except Exception as e:
        raise Exception(f"Query generation failed: {str(e)}\nRaw response: {raw_query[:200] if 'raw_query' in locals() else 'N/A'}")

def execute_query(db_type, query, connection_details):
    """Execute queries with comprehensive error handling"""
    try:
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if db_type == "mongodb":
            if not mongo_client:
                raise Exception("MongoDB connection not available")

            db = mongo_client[connection_details.get('database', 'test')]

            # Validate query format
            if not re.fullmatch(r'db\.\w+\.\w+\(.*\)', query):
                raise ValueError(f"Invalid query format. Must be 'db.collection.command()'. Got: {query[:100]}...")

            # Extract components safely
            try:
                collection_name = query.split('.')[1]
                command = query.split('.')[2].split('(')[0]
            except IndexError:
                raise ValueError("Couldn't parse collection/command from query")

            # Verify collection exists
            if collection_name not in db.list_collection_names():
                raise ValueError(f"Collection '{collection_name}' not found. Available: {db.list_collection_names()}")

            # Execute based on command type
            args_str = query.split('(', 1)[1].rsplit(')', 1)[0]
            
            if command == "find":
                try:
                    # Convert to strict JSON format
                    args_str = args_str.replace("'", '"')
                    
                    # Split into query and projection
                    if '}, {' in args_str:
                        query_part, projection_part = args_str.split('}, {', 1)
                        query_dict = json_util.loads(query_part + '}')
                        projection = json_util.loads('{' + projection_part)
                    else:
                        query_dict = json_util.loads(args_str)
                        projection = {"_id": 0}
                    
                    results = list(db[collection_name].find(query_dict, projection))
                except Exception as e:
                    raise ValueError(f"Find query parsing failed: {str(e)}\nArguments: {args_str}")

            elif command == "aggregate":
                try:
                    pipeline_str = query.split('[', 1)[1].rsplit(']', 1)[0]
                    pipeline_str = pipeline_str.replace("'", '"')
                    pipeline = json_util.loads(f'[{pipeline_str}]')
                    results = list(db[collection_name].aggregate(pipeline))
                except Exception as e:
                    raise ValueError(f"Aggregate pipeline parsing failed: {str(e)}\nPipeline: {pipeline_str}")

            else:
                raise ValueError(f"Unsupported command: {command}")

            # Serialize results
            if not results:
                return [], []
            return json.loads(json_util.dumps(results)), list(results[0].keys()) if results else ([], [])

        elif db_type == "sqlite":
            conn = None
            try:
                conn = sqlite3.connect(connection_details['database'])
                cursor = conn.cursor()
                cursor.execute(query)
                return cursor.fetchall(), [desc[0] for desc in cursor.description]
            except sqlite3.Error as e:
                return f"SQLite Error: {str(e)}", None
            finally:
                if conn:
                    conn.close()

        elif db_type == "mysql":
            engine = None
            try:
                # URL encode password for special characters
                password = quote_plus(connection_details['password'])
                
                engine = create_engine(
                    f"mysql+pymysql://{connection_details['user']}:{password}@"
                    f"{connection_details['host']}:{connection_details['port']}/"
                    f"{connection_details['database']}",
                    pool_pre_ping=True,
                    connect_args={'connect_timeout': 10}
                )
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    return result.fetchall(), result.keys()
            except Exception as e:
                return f"MySQL Error: {str(e)}", None
            finally:
                if engine:
                    engine.dispose()

        elif db_type == "postgresql":
            engine = None
            try:
                engine = create_engine(
                    f"postgresql+psycopg2://{connection_details['user']}:{connection_details['password']}@"
                    f"{connection_details['host']}:{connection_details['port']}/"
                    f"{connection_details['database']}"
                )
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    return result.fetchall(), result.keys()
            except Exception as e:
                return f"PostgreSQL Error: {str(e)}", None
            finally:
                if engine:
                    engine.dispose()

    except Exception as e:
        return f"Execution Error: {str(e)}", None

def display_results(results, columns=None):
    """Display query results in a user-friendly format"""
    if isinstance(results, str):
        st.error(results)
        return
    
    st.subheader("Query Results:")
    
    if not results:
        st.info("No results found")
        return
    
    if isinstance(results[0], dict):
        df = pd.DataFrame(results)
        st.dataframe(df)
        
        with st.expander("View Raw Documents"):
            for i, doc in enumerate(results[:3]):
                st.json(doc)
    else:
        df = pd.DataFrame(results, columns=columns) if columns else pd.DataFrame(results)
        st.dataframe(df)
    
    if not df.empty:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            "Download as Excel",
            output.getvalue(),
            "query_results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def main():
    st.set_page_config(page_title="Advanced DB Query Generator", layout="wide")
    st.title("Database Query Generator with Gemini AI")
    
    # Database type selection
    db_options = ["mysql", "postgresql", "sqlite", "mongodb"]
    db_type = st.selectbox("Select Database Type", db_options, index=0)
    
    # Get connection details
    connection_details = get_connection_details(db_type)
    
    # Special handling for MongoDB JSON upload
    if db_type == "mongodb" and 'database' not in connection_details:
        st.warning("Please either upload a JSON/BSON file or configure MongoDB connection")
        return
    
    # Only proceed if we have connection details
    if not connection_details.get('database') and db_type != "mongodb":
        st.warning("Please configure your database connection first")
        return
    
    # Query interface
    st.subheader("Query Interface")
    question = st.text_area("Enter your question about the data:", height=100)
    
    if st.button("Generate and Execute Query"):
        with st.spinner("Processing..."):
            try:
                # Get schema information
                schema_info = get_schema_info(db_type, connection_details)
                if "error" in schema_info:
                    raise Exception(schema_info["error"])
                
                # Generate query
                sample_docs = None
                if db_type == "mongodb":
                    sample_docs = [v.get('sample') for v in schema_info.values() if 'sample' in v]
                
                query = get_gemini_response(
                    question,
                    schema_info,
                    db_type,
                    sample_docs
                )
                
                st.subheader("Generated Query:")
                st.code(query, language="json" if db_type == "mongodb" else "sql")
                
                # Execute query
                results, columns = execute_query(db_type, query, connection_details)
                if isinstance(results, str):  # Error case
                    raise Exception(results)
                
                # Display results
                display_results(results, columns)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check your query and connection details")

if __name__ == "__main__":
    main()
