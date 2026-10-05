#--------
# Statistical ML
#--------

from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)

prediction = model.predict(X_test)


#--------
# Deep Learning 
#--------

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

output = model(X)