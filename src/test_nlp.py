from nlp_extractor import extract_skills, classify_role, infer_seniority

# Test skill extraction
text = """
We are looking for a Senior Data Engineer with 5+ years experience.
Must know Python, Spark, Kafka, Airflow, AWS Glue, Redshift, and dbt.
Experience with Terraform and Docker is a plus.
"""

skills = extract_skills(text)
role = classify_role("Senior Data Engineer", text)
seniority = infer_seniority("Senior Data Engineer", text)

print("Skills found:", skills)
print("Role category:", role)
print("Seniority:", seniority)