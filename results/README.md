# Resultados das experiências

- `baselines/`: avaliações base, por estratégia, janela (quando aplicável) e imputação.
- `tuning/configured_sliding_batch_2026-11-11/`: lote de 16 experiências, por janela configurada (2 ou 3 anos) e modelo.
- `tuning/configured_batch_2026-11-11_22-52-30-578742/`: novo lote de 16 experiências (22:52–23:19), por estratégia, janela configurada e modelo; separado do lote anterior.
- `tuning/earlier_attempts/`: tuning anterior, separado em `complete`, `incomplete` e `empty`.
- `sliding_tuning_analysis/`: relatório, CSVs, figura e script de análise.

Os nomes originais das pastas e o conteúdo dos 166 ficheiros das experiências foram preservados. A pasta `1st` fica nas avaliações expanding com imputação `unknown`: não guarda configuração suficiente para identificar a imputação.

**Nota:** as janelas do lote de tuning descrevem a configuração guardada. A análise identificou indícios de validação externa expanding. Consultar [o relatório](sliding_tuning_analysis/analysis_report.md) antes de interpretar o lote como comparação entre janelas externas.

O [índice de experiências](experiment_index.csv) contém os caminhos antigos e novos, modelo, estratégia configurada, janela, imputação e estado das 50 pastas. `organization_manifest.json` regista os caminhos e hashes SHA-256 para verificar a integridade. `organize_results.py` preserva o índice existente e organiza novos resultados; `--dry-run` permite consultar os destinos antes de mover.

A análise em `sliding_tuning_analysis/` refere-se apenas ao primeiro lote. O novo lote foi organizado, sem recalcular a análise nem assumir que o problema identificado no lote anterior também se aplica a este.

O script `sliding_tuning_analysis/analyze.py` procura também em subpastas. Os caminhos em `source_manifest.json` foram atualizados.

`paths.results` e a pipeline não foram alterados. Os próximos runs continuam a criar pastas diretamente em `results/`; esta organização abrange os resultados já existentes. Nenhum treino foi executado.
