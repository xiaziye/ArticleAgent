#!/usr/bin/env python3
"""
Export OpenAlex paper data from PostgreSQL to JSON format.

Features:
- Efficient batched export using OFFSET/LIMIT
- Filters abstracts (non-empty) and keywords (score > 0.1)
- Generates valid JSON array (no trailing commas)
- Memory-safe for large datasets

Usage:
    python export_papers_to_json.py --output nsu_papers_keywords.json
"""

import argparse
import json
import psycopg2
from tqdm import tqdm


# Database configuration (consider moving to config file in production)
DB_CONFIG = {
    "dbname": "openalex_db",
    "user": "postgres",
    "password": "Xzy011230*",
    "host": "localhost",
    "port": "5432"
}


def export_to_json(output_file: str, batch_size: int = 1000) -> None:
    """
    Export papers with non-empty abstracts and high-confidence keywords to JSON.

    Args:
        output_file: Path to output JSON file
        batch_size: Number of records to fetch per database query
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        # Get total count for progress bar
        count_query = """
        SELECT COUNT(*)
        FROM works_csv
        WHERE abstract IS NOT NULL 
          AND abstract != ''
          AND keywords IS NOT NULL
        """
        cursor.execute(count_query)
        total = cursor.fetchone()[0]

        if total == 0:
            print("⚠️ No records found matching the criteria.")
            return

        # Open output file and write JSON array start
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')

        offset = 0
        first_record = True

        with tqdm(total=total, desc="Exporting papers") as pbar:
            while True:
                # Fetch batch with keyword filtering in SQL
                data_query = """
                SELECT id, title, abstract, filtered_keywords
                FROM (
                    SELECT 
                        id,
                        title, 
                        abstract,
                        (SELECT jsonb_agg(k ORDER BY (k->>'score')::float DESC)
                         FROM jsonb_array_elements(keywords) k 
                         WHERE (k->>'score')::float > 0.1) AS filtered_keywords
                    FROM works_csv 
                    WHERE abstract IS NOT NULL 
                      AND abstract != ''
                      AND keywords IS NOT NULL
                ) AS subquery
                WHERE filtered_keywords IS NOT NULL
                ORDER BY id
                LIMIT %s OFFSET %s;
                """
                cursor.execute(data_query, (batch_size, offset))
                rows = cursor.fetchall()

                if not rows:
                    break

                # Append records to JSON file
                with open(output_file, 'a', encoding='utf-8') as f:
                    for row in rows:
                        if not first_record:
                            f.write(',\n')
                        else:
                            first_record = False

                        paper = {
                            "id": row[0],
                            "title": row[1] or "",  # Handle NULL titles
                            "abstract": row[2],
                            "keywords": row[3] if row[3] else []
                        }
                        f.write(json.dumps(paper, ensure_ascii=False, indent=2))

                offset += len(rows)
                pbar.update(len(rows))

        # Close JSON array
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write('\n]')

        print(f"\n✅ Successfully exported {offset} papers to {output_file}")

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OpenAlex papers to JSON")
    parser.add_argument("--output", default="nsu_papers_keywords.json", help="Output JSON file path")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for DB queries")
    args = parser.parse_args()

    export_to_json(args.output, args.batch_size)
