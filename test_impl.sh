uv run python3 src/orchestrator.py # > /dev/null
./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/dataset_docs_public.json data/datasets/public/AnsweredQuestions/dataset_docs_public.json --k 10
./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/dataset_code_public.json data/datasets/public/AnsweredQuestions/dataset_code_public.json --k 10
