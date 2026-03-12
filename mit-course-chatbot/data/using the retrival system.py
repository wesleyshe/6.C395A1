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
