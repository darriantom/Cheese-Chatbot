@echo off
echo Starting daily update...
python scraper.py
python ingest.py
python chatbot.py