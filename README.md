# Perceptron & Pegasos

Implementação de dois classificadores de emails usando [Perceptron](https://en.wikipedia.org/wiki/Perceptron) e SVM [Pegasos](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf). O dataset utilizado é subconjunto do SpamAssassin Public Corpus.

## Dataset

Os arquivos de input estão separados da seguinte forma:
* spam_train.txt para o treino
* spam_val.txt para a validação

Cada linha é um email, começando com o rótulo 1 ou 0, 1 para spam e 0 não-spam. Os emails estão normalizados (URL's, valores monetários, endereços de emails), para melhorar o desempenho do classificador.

## Execução

```
python main.py
```

O número de iterações máxima do perceptron pode ser alterada pela constante PERCEPTRON_MAX_ITER, assim como quais lambdas o Pegasos irá executar.

## Output

Ao final da execução, é mostrado a % de erros ao executar no dataset de validação.
