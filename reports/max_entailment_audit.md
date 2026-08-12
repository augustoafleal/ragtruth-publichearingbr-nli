# Auditoria read-only do baseline `max-entailment`

## Classificação final

**BASELINE NOT READY FOR PAIRED COMPARISON**  
**DO NOT IMPLEMENT BOOTSTRAP YET**

Os valores `AUPRC=0,343` e `AUROC=0,769` aparecem no repositório apenas como referências descritivas hardcoded. Não foi localizada uma implementação executável do baseline, um manifesto de execução, um checkpoint/tokenizer, ou previsões individuais por exemplo.

## Fonte dos números

As ocorrências canônicas encontradas foram:

- `src/ragtruth_transfer/ragtruth_zero_shot.py:372`: dicionário `baseline_max_entailment`, marcado como `descriptive_reference_from_protocol`;
- `src/ragtruth_transfer/ragtruth_confirmatory.py:355`: os mesmos valores no agregador confirmatório;
- `docs/experiments/ragtruth_broad_epoch1_publichearing_transfer.md:54,58`: tabela narrativa com Precision, Recall, F1, FPR, MCC, AUPRC e AUROC;
- `proximos_passos.md`: apenas plano futuro de comparação pareada.

O histórico Git mostra que os números foram introduzidos como constantes de referência; não há commit que acrescente código ou previsões do `max-entailment`.

Não existe uma fonte canônica de dados do baseline no checkout. Portanto, não é possível afirmar que esses números foram produzidos pelos mesmos 4.235 exemplos da campanha atual.

## Definição e modelo NLI

Não foi encontrada implementação do baseline PublicHearingBR chamada `max-entailment`. Assim, permanecem desconhecidos:

- `model_id`, revisão, tokenizer e arquitetura usados pelo baseline;
- premise/hypothesis, separadores, truncamento, padding e normalização;
- índices de entailment, neutral e contradiction;
- fórmula de conversão dos logits;
- agregação entre chunks;
- orientação do score final;
- tratamento de slots inválidos e exemplos com menos evidências;
- qualquer treinamento, calibração ou seleção de threshold no PublicHearingBR.

Existe uma implementação relacionada, mas diferente, em `src/ragtruth_transfer/support_verification.py`. Ela usa o modelo `MoritzLaurer/ernie-m-base-mnli-xnli` (revisão documentada `6ae6ab63e35c78f3700d64479f43e203385a1ecc`), localiza a classe `entailment` pelo `id2label`, aplica `softmax` e calcula `P(entailment)` para pares evidência → claim. Para os negativos Broad, também calcula `individual_max_score=max(P(entailment))` e um score de concatenação. Esse módulo é um verificador Strict do RAGTruth; não é uma execução do baseline PublicHearingBR e não fornece as previsões usadas nos números `0,343/0,769`.

## Dados e pareamento

A campanha confirmatória atual possui, para cada seed, 4.235 linhas, 4.235 `example_id` únicos, 206 `hearing_id`, 501 positivos, 3.734 negativos e prevalência `0,1182998819`. As previsões estão em:

- `runs/ragtruth_confirmatory/4e12933c51136624/seed_0/publichearing_zero_shot/6f6d3122e3c107cc/predictions.parquet`;
- `runs/ragtruth_confirmatory/4e12933c51136624/seed_1/publichearing_zero_shot/a6ca16eb7e681425/predictions.parquet`;
- `runs/ragtruth_confirmatory/4e12933c51136624/seed_2/publichearing_zero_shot/2e994a84d0469e56/predictions.parquet`.

Essas previsões têm score contínuo `probability`, labels e identificadores. Porém, não existe arquivo equivalente do baseline com `example_id`, `hearing_id` e score contínuo. Consequentemente, não foi possível executar o join nem medir IDs ausentes, labels divergentes, audiências divergentes, duplicatas ou cobertura pareável. A comparação entre os 4.235 exemplos e os números antigos de um documento que menciona 4.237 exemplos não pode ser confirmada.

## Reprodução das métricas

Não foi possível recalcular AUPRC, AUROC ou Brier do baseline: não há scores individuais. Os valores registrados são reproduzíveis apenas como leitura das constantes, não como resultado de uma avaliação independente. Não há tolerância numérica aplicável nem evidência sobre orientação do score.

## Thresholds

O documento antigo registra F1, FPR e MCC, mas não registra o valor, a origem ou a regra de seleção do threshold. Portanto:

- **threshold-free comparison readiness:** não pronto;
- **thresholded comparison readiness:** não pronto.

Os scores threshold-free da campanha RAGTruth → PublicHearingBR não podem ser comparados por bootstrap pareado ao baseline sem previsões do baseline alinhadas por ID. F1/MCC/FPR exigiriam, adicionalmente, threshold com proveniência verificável.

## Arquivos inspecionados

- `src/ragtruth_transfer/ragtruth_zero_shot.py`;
- `src/ragtruth_transfer/ragtruth_confirmatory.py`;
- `src/ragtruth_transfer/support_verification.py`;
- manifests, previsões e métricas em `runs/ragtruth_confirmatory/4e12933c51136624/`;
- resultados e arquivos de `results/` relacionados a PublicHearingBR.

## Comandos e garantias

Foram usados somente buscas (`rg`, `find`), leitura (`sed`, `git show`, `git blame`, `git log`) e scripts Python em memória para ler JSON/CSV/Parquet/ZIP e verificar IDs, labels, contagens, hashes e schemas. Não foram baixados modelos, carregados checkpoints, executados treinamentos, geradas previsões, calculados bootstraps ou testes de superioridade.

Nenhum arquivo científico, dataset, previsão, métrica, hash ou manifesto foi alterado. Este relatório é o único artefato criado nesta auditoria. Nenhum commit foi feito.

## Decisão

Os números `0,343` e `0,769` devem ser tratados como **referência descritiva não auditável**, e não como baseline pareável. Para liberar o bootstrap pareado, é necessário obter do experimento original o manifesto/configuração do `max-entailment` e um arquivo de previsões contínuas contendo, no mínimo, `example_id`, `hearing_id`, `label` e score final por exemplo.
