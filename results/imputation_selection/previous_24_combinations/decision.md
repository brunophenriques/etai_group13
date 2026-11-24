# Decisao de imputacao

- **classification: median_0_uniform__unknown**. F1 medio = 0.715440; melhor valor observado = 0.718827 (knn_10_uniform__unknown).
- **regression: median_0_uniform__unknown**. MAE medio = 1438.552329; melhor valor observado = 1436.809923 (knn_3_distance__unknown).

## Protocolo

24 combinacoes, quatro folds temporais (2020-2023), janela de tres anos e embargo de 90 dias. Classificacao: regressao logistica e random forest, threshold 0.5. Regressao condicional: Ridge e random forest; so regressados com gasto observado fornecem targets. O preprocessing nao supervisionado usa todos os preditores do treino de cada fold.

F1 e MAE medios atribuem igual peso a cada fold e modelo. Entre candidatos ate 0.005 F1 ou 1% MAE do melhor, escolhemos o mais simples. Sao tolerancias praticas, nao significancia estatistica. A dispersao na tabela combina modelos e folds; scores.csv permite inspecionar cada componente.

## Regras comuns

Limpeza de valores invalidos, depois preenchimento estrutural e recuperacao, depois imputacao. A ultima transacao anterior disponivel e assumida como a compra anterior verdadeira. Historico limitado ao treino, estritamente anterior a cada compra, sem usar outras linhas de teste. Os zeros estruturais nao sao doadores de historicos desconhecidos. Indicators de ausencia original, falta de historico, correcao e recuperacao preservam a informacao. Numericas fora das colunas multivariadas usam mediana elegivel; sem doadores usa-se zero com indicador. Datas desconhecidas nao sao inventadas: ano/mes 0, dia da semana -1 e indicador de data ausente.

## Limites da evidencia

5634 linhas de desenvolvimento datadas. 65 linhas sem data no train ficam fora da avaliacao temporal. 119 regressados no desenvolvimento nao tem gasto conhecido. unknown_spend_profile.csv compara os seus perfis com os restantes regressados. Nao imputamos targets nem contamos estes gastos desconhecidos nas metricas. A previsao preserva linhas sem data e clientes novos com fallbacks aprendidos no treino.

Os valores aqui sao resultados de selecao, nao uma estimativa independente de desempenho final. 2024, anos posteriores e teste nao participaram nesta escolha; o EDA anterior do grupo ja tinha observado o train completo. A ocultacao artificial mede um mecanismo sintetico e nao prova a qualidade nos missing reais. A escolha deve ser reavaliada se mudarem features ou modelos.

## Artefactos e reproducao

- scores.csv: metricas e tempos por combinacao, modelo e fold.
- selection_summary.csv / selected_config.yaml: criterio e configuracoes escolhidas.
- masked_diagnostics.csv: recuperacao de 10% de valores conhecidos, seeds 42, 43 e 44.
- missingness_audit.csv / missingness_by_year.csv / missingness_patterns.csv: diagnostico.
- end_to_end.csv: quatro pares classificador/regressor, incluindo falsos positivos e negativos.
- selected_oof_predictions.csv: previsoes de validacao dos pares selecionados.
- warnings.csv: avisos capturados, incluindo eventual falta de convergencia do iterative.
- config_snapshot.yaml / metadata.json: configuracao, versoes e fingerprint dos dados.

```bash
python -m src.imputation_experiment --output results/imputation_selection_reproduction
python -m src.imputation_report --results results/imputation_selection_reproduction
python -m tests
```

A pasta de resultados deve ser nova para evitar misturar experiencias. O notebook le os resultados guardados; a experiencia integral corre pelo comando acima. Usar --apply-config no gerador do relatorio para aplicar as escolhas ao config.yaml.
