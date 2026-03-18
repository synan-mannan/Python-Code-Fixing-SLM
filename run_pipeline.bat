@echo off
echo Running dataset pipeline...
python -m data_pipeline.data_scraper --pages 5
python -m data_pipeline.github_issue_scraper --pages 5
python data_pipeline/data_cleaner.py
python data_pipeline/dataset_formatter.py
echo Pipeline complete! Check dataset/
pause

