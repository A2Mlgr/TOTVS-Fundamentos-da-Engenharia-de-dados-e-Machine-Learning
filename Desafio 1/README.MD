#  Transfer Learning — Cats vs Dogs

Projeto desenvolvido como parte do desafio da **DIO (Digital Innovation One)**, aplicando **Transfer Learning** com TensorFlow/Keras para classificação binária de imagens de gatos e cachorros.


O objetivo é demonstrar na prática o uso de Transfer Learning em Deep Learning, reutilizando uma rede neural pré-treinada (MobileNetV2 no ImageNet) e adaptando-a para um novo problema de classificação de imagens.

O treinamento é dividido em duas fases:

1. **Fase 1 — Feature Extraction:** o backbone da MobileNetV2 é congelado e apenas as camadas do topo (classificador) são treinadas.
2. **Fase 2 — Fine-Tuning:** as últimas 30 camadas do backbone são descongeladas e retreinadas com uma taxa de aprendizado menor, refinando os pesos para o novo domínio.

---

## Estrutura do Projeto

```
.
├── transfer_learning.py        # Script principal
├── kagglecatsanddogs_5340.zip  # Dataset original (Kaggle)
├── PetImages/                  # Dataset extraído automaticamente
│   ├── Cat/
│   └── Dog/
├── best_model.keras            # Melhor modelo salvo durante o treino
├── training_curves.png         # Gráfico de acurácia e loss
└── sample_predictions.png      # Grade com predições de amostra
```

---

## Arquitetura

| Componente         | Detalhe                          |
|--------------------|----------------------------------|
| Backbone           | MobileNetV2 (pré-treinado ImageNet) |
| Input              | 160 × 160 × 3                   |
| Pooling            | GlobalAveragePooling2D           |
| Dense              | 128 unidades, ReLU               |
| Dropout            | 0.3 → 0.2                       |
| Output             | 1 unidade, Sigmoid (binário)     |
| Loss               | Binary Crossentropy              |
| Otimizador Fase 1  | Adam (lr = 1e-3)                 |
| Otimizador Fase 2  | Adam (lr = 1e-5)                 |

---

## Requisitos

- Python 3.8+
- TensorFlow 2.x
- Matplotlib

Instale as dependências com:

```bash
pip install tensorflow matplotlib
```

---

## Como Executar

1. Certifique-se de que o arquivo `kagglecatsanddogs_5340.zip` está na mesma pasta que o script.

2. Execute o script:

```bash
python desafio_1ML.py
```

O script irá automaticamente:
- Extrair o dataset do ZIP
- Remover imagens corrompidas
- Treinar o modelo em duas fases
- Salvar o melhor modelo em `best_model.keras`
- Gerar os gráficos de treino e predições de amostra

---

## Resultados Esperados

Após o treinamento completo, espera-se atingir uma acurácia de validação acima de **95%**, graças ao aproveitamento dos pesos pré-treinados do ImageNet e ao processo de fine-tuning.

Os arquivos gerados ao final da execução são:

- `best_model.keras` — modelo com melhor desempenho na validação
- `training_curves.png` — evolução de acurácia e loss por época
- `sample_predictions.png` — grade 3×3 com predições visuais

---

## Dataset

- **Nome:** Microsoft Cats and Dogs Dataset
- **Fonte:** [Kaggle](https://www.kaggle.com/datasets/microsoft/catsvsdogs)
- **Classes:** `Cat` | `Dog`
- **Total de imagens:** ~25.000 (após remoção de corrompidas)

---

## 🔗 Referências

- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [DIO — Digital Innovation One](https://www.dio.me)