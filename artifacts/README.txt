Размещение артефактов RuBERT→music:

- artifacts/runs → символическая ссылка на каталог rubert_runs в корне репозитория.
- После новых прогонов обновите manifest.jsonl (поля output_dir, status).
- Сводка для LaTeX: python3 scripts/aggregate_run_metrics.py --main-protocol-only
  → artifacts/summary.csv и coursework-latex/lyrata-kurs/tables/results_generated.tex
- Диаграмма: python3 scripts/plot_rubert_retrieval_hit_rates.py
