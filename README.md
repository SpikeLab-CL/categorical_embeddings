## Categorical Embeddings
**IMPORTANT:** Still under construction

This packages allows to create embeddings from categorical variables using Keras,
you must specify `target_type` which must be one of these: 
1. `regression`
2. `binary_classification`
3. `multiclass`

### Instalation
Run `python setup.py install`

### Usage

Basic usage example:
```
import pandas as pd
from categorical_embeddings import embedder

data = pd.read_csv("Pokemon.csv")
data.fillna("NA", inplace=True)
data = data[['Type_1', 'Type_2', 'Total', 'Legendary']]
data.head()

   Type_1  Type_2  Total  Legendary
0  Grass  Poison    318      False
1  Grass  Poison    405      False
2  Grass  Poison    525      False
3  Grass  Poison    625      False
4   Fire      NA    309      False

emb = embedder.Embedder(target_type="regression")
data = emb.fit_transform(data, y="Total")

#you can exclude columns from the embedding
#data = emb.fit_transform(data, y="Total",exclude_columns=['col_to_exlude])

data.head().T

                     0            1            2            3           4
Type_1           Grass        Grass        Grass        Grass        Fire
Type_2          Poison       Poison       Poison       Poison          NA
Total              318          405          525          625         309
Legendary        False        False        False        False       False
Type_1_1     0.0243019    0.0243019    0.0243019    0.0243019 -0.00465042
Type_1_2     0.0112878    0.0112878    0.0112878    0.0112878  -0.0400655
Type_1_3   0.000585036  0.000585036  0.000585036  0.000585036   0.0743753
Type_1_4    -0.0476271   -0.0476271   -0.0476271   -0.0476271   0.0179989
Type_1_5     0.0216034    0.0216034    0.0216034    0.0216034 -0.00587115
Type_1_6    -0.0126848   -0.0126848   -0.0126848   -0.0126848  -0.0696179
Type_1_7   -0.00864204  -0.00864204  -0.00864204  -0.00864204   0.0261521
Type_1_8    -0.0346298   -0.0346298   -0.0346298   -0.0346298  -0.0579275
Type_1_9     0.0157211    0.0157211    0.0157211    0.0157211   0.0704117
Type_2_1     0.0458701    0.0458701    0.0458701    0.0458701  -0.0578474
Type_2_2    -0.0306355   -0.0306355   -0.0306355   -0.0306355   0.0836497
Type_2_3    -0.0661783   -0.0661783   -0.0661783   -0.0661783   0.0824363
Type_2_4    -0.0200662   -0.0200662   -0.0200662   -0.0200662   0.0520892
Type_2_5    -0.0318695   -0.0318695   -0.0318695   -0.0318695   0.0858483
Type_2_6    -0.0138621   -0.0138621   -0.0138621   -0.0138621   0.0187802
Type_2_7    -0.0586479   -0.0586479   -0.0586479   -0.0586479   0.0693317
Type_2_8     0.0102438    0.0102438    0.0102438    0.0102438  -0.0244231
Type_2_9   -0.00574593  -0.00574593  -0.00574593  -0.00574593  -0.0559954
Type_2_10   -0.0390857   -0.0390857   -0.0390857   -0.0390857  -0.0102417

#if you want only the components
embeddings_values = emb.fit(data['Type_1'], y=data["Total"])
embeddings_values.head()

   Type_1_1  Type_1_2  Type_1_3  Type_1_4  Type_1_5  Type_1_6  Type_1_7  Type_1_8  Type_1_9    Type_1
0  0.024256 -0.070111 -0.000073  0.025817 -0.052308  0.059688  0.027244  0.043765 -0.000608       Bug
1 -0.065904  0.017269  0.003565  0.014029  0.043661 -0.023009  0.037779  0.008503 -0.053083      Dark
2 -0.046566  0.087611  0.087385  0.064650  0.089642 -0.056577  0.119586  0.100826 -0.119916    Dragon
3 -0.024248  0.083361  0.047766  0.008144  0.021174 -0.041157  0.004702 -0.071176 -0.040287  Electric
4 -0.053012 -0.017219  0.028762  0.058204 -0.024770  0.019870 -0.027558  0.008474 -0.040889     Fairy
```

