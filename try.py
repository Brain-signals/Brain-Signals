from vape_model.files import open_dataset
from vape_model.model import initialize_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder

X1,y1 = open_dataset('MRI_PD_vanicek_control', verbose=1, limit=20)
X2,y2 = open_dataset('MRI_PD_vanicek_parkinsons', verbose=1, limit=50)
X3,y3 = open_dataset('Wonderwall_alzheimers', verbose=1, limit=50)
X4,y4 = open_dataset('Wonderwall_control', verbose=1, limit=15)
X5,y5 = open_dataset('MRI_PD_1_control', verbose=1, limit=15)
X6,y6 = open_dataset('MRI_PD_1_parkinsons', verbose=1, limit=50)


X = np.concatenate((X1,X2,X3,X4,X5,X6))
y = pd.concat((y1,y2,y3,y4,y5,y6),ignore_index=True)

enc = OneHotEncoder(sparse = False)
enc.fit(y[['diagnostic']])
y_encoded = enc.transform(y[['diagnostic']]).astype('int8')

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.30)

target_res = int(os.environ.get("TARGET_RES"))

model=initialize_model(target_res, target_res, target_res)

es = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)

# Train the model, doing validation at the end of each epoch
history=model.fit(
    X_train, y_train,
    validation_split=0.3,
    epochs=100,
    verbose=1,
    callbacks=[es],
)

print(model.evaluate(X_test, y_test))
