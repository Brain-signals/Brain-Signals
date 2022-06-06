from vape_model.files import open_dataset
from vape_model.model import initialize_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.callbacks import EarlyStopping

X1,y1 = open_dataset('MRI_PD_vanicek_control',verbose=1)
X2,y2 = open_dataset('MRI_PD_vanicek_parkinsons',verbose=1)

X = np.concatenate((X1,X2))
y = pd.concat((y1,y2),ignore_index=True)

y_encoded=np.array([ 0 if i=='Healthy' else 1 for i in y['diagnostic']])

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
