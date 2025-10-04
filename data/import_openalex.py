#!/usr/bin/env python3
"""
Optimized OpenAlex CSV Importer for Academic Knowledge Graph Construction

This script imports OpenAlex `works.csv` into a PostgreSQL database with:
- Robust ID normalization (handles URLs, trailing spaces, malformed formats)
- Abstract cleaning and filtering (min length, HTML removal, prefix stripping)
- Authorship parsing into structured JSONB
- Incremental & full import support
- Duplicate detection via data hashing
- Detailed logging and import history tracking

Designed for reproducibility in academic research (e.g., concept path mining).

Usage:
    python data/import_openalex.py \
        --csv data/raw/works.csv \
        --dbname academic_kg \
        --user your_user \
        --password your_password \
        --import-type incremental
"""

import csv
import json
import psycopg2
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import re
import os
import hashlib


# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/openalex_import.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_work_id(openalex_id: str) -> str | None:
    """
    Extract clean OpenAlex work ID (e.g., 'W123456') from various input formats.

    Handles:
    - Full URLs: 'https://openalex.org/W100109915    '
    - Raw IDs: 'W100109915'
    - Numeric-only: '100109915'
    - Malformed strings with trailing whitespace or special chars

    Returns:
        Cleaned work ID string (e.g., 'W100109915') or None if invalid.
    """
    if not openalex_id:
        return None

    # Normalize: strip whitespace and newlines
    openalex_id = str(openalex_id).strip()

    # Case 1: Full OpenAlex URL
    if "openalex.org" in openalex_id:
        work_id = openalex_id.split("/")[-1].strip()
        work_id = re.sub(r'[^\w]', '', work_id)  # Remove non-alphanumeric chars
        return work_id

    # Case 2: Already in W-prefixed format
    if openalex_id.startswith("W"):
        return re.sub(r'[^\w]', '', openalex_id)

    # Case 3: Pure numeric ID
    if openalex_id.isdigit():
        return f"W{openalex_id}"

    # Case 4: Fallback regex match for W + digits
    match = re.search(r'W\d+', openalex_id)
    return match.group(0) if match else None


def clean_abstract(abstract: str, min_length: int = 50) -> str | None:
    """
    Clean and validate abstract text for academic use.

    Steps:
    1. Remove HTML tags
    2. Collapse whitespace
    3. Strip common prefixes like "Abstract:"
    4. Enforce minimum length (default: 50 chars)

    Returns:
        Cleaned abstract string or None if invalid/short.
    """
    if not abstract:
        return None

    # Remove HTML tags
    abstract = re.sub(r'<[^>]+>', '', abstract)
    # Normalize whitespace
    abstract = re.sub(r'\s+', ' ', abstract).strip()
    # Remove common prefixes
    abstract = re.sub(r'^[Aa]bstract[:\s]*', '', abstract)

    return abstract if len(abstract) >= min_length else None


def parse_authors(authorships_str: str) -> str | None:
    """
    Parse OpenAlex authorships field into structured JSON.

    Input format: pipe-separated JSON objects or raw strings.
    Output: JSON string of list of authors with id, display_name, position.

    Returns:
        JSON string or None if parsing fails.
    """
    if not authorships_str:
        return None

    authors = []
    try:
        # Split multiple author entries
        entries = authorships_str.split("|")
        for entry in entries:
            entry = entry.strip()
            if entry.startswith('{') and entry.endswith('}'):
                try:
                    data = json.loads(entry)
                    if isinstance(data, dict) and "author" in data:
                        author = data["author"]
                        if author.get("display_name"):
                            authors.append({
                                "id": author["id"].split("/")[-1].strip() if author.get("id") else None,
                                "display_name": author["display_name"].strip(),
                                "position": data.get("author_position", "unknown")
                            })
                except json.JSONDecodeError:
                    continue

        if authors:
            return json.dumps(authors)

        # Fallback: extract raw names if JSON fails
        if "raw_author_name" in authorships_str:
            names = re.findall(r'"raw_author_name":\s*"([^"]+)"', authorships_str)
            authors = [{"id": None, "display_name": name.strip(), "position": "unknown"} for name in names]
            return json.dumps(authors) if authors else None

    except Exception as e:
        logger.debug(f"Author parsing error: {e}")

    return None


def calculate_data_hash(row: dict) -> str:
    """
    Compute MD5 hash of key fields to detect duplicate imports.
    Used to avoid reprocessing identical records.
    """
    key_data = (
        row.get("id", "") +
        (row.get("title", "") or "") +
        (row.get("abstract", "") or "")[:100]
    )
    return hashlib.md5(key_data.encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Import OpenAlex works.csv into PostgreSQL")
    parser.add_argument('--csv', required=True, help='Path to OpenAlex works.csv')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    parser.add_argument('--dbname', required=True, help='Database name')
    parser.add_argument('--user', required=True, help='Database user')
    parser.add_argument('--password', required=True, help='Database password')
    parser.add_argument('--batch-size', type=int, default=500, help='Batch size for DB inserts')
    parser.add_argument('--import-type', choices=['full', 'incremental'], default='incremental',
                        help='Import mode: full (replace) or incremental (skip duplicates)')
    parser.add_argument('--min-abstract-length', type=int, default=50,
                        help='Minimum abstract length to retain')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()
    os.makedirs("logs", exist_ok=True)

    start_time = datetime.now()
    logger.info(f"Starting OpenAlex import (CSV: {args.csv}, mode: {args.import_type})")

    # Connect to DB
    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password
    )
    conn.set_client_encoding('UTF8')
    cursor = conn.cursor()

    try:
        # Create import history record
        cursor.execute("""
            INSERT INTO import_history (import_type, start_time, status, csv_file)
            VALUES (%s, %s, 'started', %s) RETURNING id
        """, (args.import_type, start_time, args.csv))
        import_id = cursor.fetchone()[0]
        conn.commit()

        # Process CSV
        with open(args.csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            total_processed = 0
            total_imported = 0

            for row in tqdm(reader, desc="Processing works"):
                # Extract and validate ID
                work_id = extract_work_id(row.get("id", ""))
                if not work_id:
                    continue

                # Skip duplicates
                data_hash = calculate_data_hash(row)
                cursor.execute("SELECT 1 FROM import_details WHERE work_id = %s AND data_hash = %s",
                               (work_id, data_hash))
                if cursor.fetchone():
                    continue

                # Clean abstract
                cleaned_abstract = clean_abstract(row.get("abstract", ""), args.min_abstract_length)

                # Parse other fields
                doi = row.get("doi", "").strip()
                if doi.startswith("https://doi.org/"):
                    doi = doi.replace("https://doi.org/", "").strip() or None

                title = row.get("title", "").strip() or None
                display_name = row.get("display_name", "").strip() or None
                publication_year = int(row["publication_year"]) if row.get("publication_year", "").isdigit() else None
                authors = parse_authors(row.get("authorships", ""))

                # Insert or update
                cursor.execute("SELECT 1 FROM works_csv WHERE id = %s", (work_id,))
                exists = cursor.fetchone()

                if exists:
                    cursor.execute("""
                        UPDATE works_csv SET
                            doi = %s, title = %s, display_name = %s, publication_year = %s,
                            abstract = %s, authors = %s, updated_date = %s
                        WHERE id = %s
                    """, (doi, title, display_name, publication_year,
                          cleaned_abstract, authors, row.get("updated_date"), work_id))
                else:
                    cursor.execute("""
                        INSERT INTO works_csv (
                            id, doi, title, display_name, publication_year,
                            abstract, authors, updated_date
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (work_id, doi, title, display_name, publication_year,
                          cleaned_abstract, authors, row.get("updated_date")))

                # Log import detail
                cursor.execute("""
                    INSERT INTO import_details (import_id, work_id, data_hash, abstract_imported, import_time, status)
                    VALUES (%s, %s, %s, %s, %s, 'success')
                """, (import_id, work_id, data_hash, bool(cleaned_abstract), datetime.now()))

                total_imported += 1
                total_processed += 1

                if total_imported % args.batch_size == 0:
                    conn.commit()

        # Final commit and stats
        conn.commit()
        cursor.execute("SELECT calculate_abstract_stats()")
        cursor.execute("""
            UPDATE import_history SET
                end_time = %s, status = 'completed',
                total_processed = %s, total_imported = %s
            WHERE id = %s
        """, (datetime.now(), total_processed, total_imported, import_id))
        conn.commit()

        logger.info(f"✅ Import completed: {total_imported}/{total_processed} records processed.")

    except Exception as e:
        logger.exception(f"Import failed: {e}")
        if 'import_id' in locals():
            cursor.execute("""
                UPDATE import_history SET end_time = %s, status = 'failed', error_message = %s WHERE id = %s
            """, (datetime.now(), str(e), import_id))
            conn.commit()
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
