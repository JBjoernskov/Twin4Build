import psycopg2

# Database connection
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='postgres',
    user='postgres',
    password='password'
)

cursor = conn.cursor()

# Query to find available UUIDs
cursor.execute("""
    SELECT DISTINCT uuid, COUNT(*) as data_points
    FROM bts_site_b
    WHERE uuid IS NOT NULL
    GROUP BY uuid
    ORDER BY data_points DESC;
""")

results = cursor.fetchall()

print(f"Found {len(results)} UUIDs with data:")
print("\nUUID                                  | Data Points")
print("-" * 50)

for uuid, count in results:
    print(f"{uuid:<36} | {count:>10}")

cursor.close()
conn.close()
