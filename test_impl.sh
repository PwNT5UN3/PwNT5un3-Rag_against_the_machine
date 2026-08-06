uv run python3 src/orchestrator.py # > /dev/null
./moulinette/moulinette-ubuntu evaluate_student_search_results docs.json datasets_public/public/AnsweredQuestions/dataset_docs_public.json --k 10 | grep 'recall@5'
./moulinette/moulinette-ubuntu evaluate_student_search_results code.json datasets_public/public/AnsweredQuestions/dataset_code_public.json --k 10 | grep 'recall@5'
