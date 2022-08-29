### External imports ###

import time

### Internal imports ###

from brainsignals.evaluate import evaluate_model
from brainsignals.model_class import Model
from brainsignals.preprocess_class import Preprocessor
from brainsignals.utils import time_print


### Settings ###

chosen_datasets = [('Controls', 50), # max = 63
                   ('MRI_PD_vanicek_control', 20), # max = 21
                   ('MRI_PD1_control', 10), # max = 15
                   ('Wonderwall_control', 70), # max = 424

                   #('MRI_PD1_parkinsons', 30), # max = 30
                   #('MRI_PD_vanicek_parkinsons', 20), # max = 20

                   ('Wonderwall_alzheimers', 150), # max = 197
                  ]

target_res = 128
slicing_bot = 0.4
slicing_top=0.1
dataset_verbose=2

learning_rate=0.0002
epochs=100
patience=20
monitor='val_accuracy'
batch_size=32
training_verbose=1

max_run=20


### Functions ###

def preprocess_and_train():
    my_modele = Model()
    my_preproc = Preprocessor()

    my_preproc.initialize_preprocessor(target_res=target_res,
                                       slicing_bot=slicing_bot,
                                       slicing_top=slicing_top)
    X, y = my_preproc.create_dataset(chosen_datasets, verbose=dataset_verbose)

    my_modele.initialize_model(my_preproc,
                               learning_rate=learning_rate)
    my_modele.train_model(X, y,
                        epochs=epochs,
                        patience=patience,
                        monitor=monitor,
                        batch_size=batch_size,
                        verbose=training_verbose)

    my_modele.save_model()

    evaluate_model(my_modele.model_id, max_run=max_run)

    return my_modele.model_id


### Launch ###

if __name__ == '__main__':
    start = time.perf_counter()
    model_id = preprocess_and_train()
    end = time.perf_counter()
    print(f'Model {model_id} has been created and evaluated in',time_print(start,end))
