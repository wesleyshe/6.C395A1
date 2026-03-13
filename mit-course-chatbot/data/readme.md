- Departments is always a list, even for single-department courses — account for this in prompts
- Instructors are mostly empty (~97%). This is because the catalog HTML makes them hard to parse reliably. I considered cross scrapping with each department page but way too complex and time consuming.
- I used "CI-HW" not "CI-H" for the filter 
- Python 3.9

- to use the retrieval.search:

python
from retrieval.search import search_courses
results = search_courses(
    query="machine learning for biology",
    filters={
        "department": "Biology",
        "distribution_requirement": "REST",
        "has_prereqs": False,
        "keyword": "6.100",
    },
    top_k=10
)
