#!/bin/bash

python3 optuna_lightning_model.py --img_size=624 --real_test --log_name=img_size_trials/test --limit_train_batches=0.1 --limit_val_batches=0.1 --n_trials=1 --epochs=1

python3 optuna_lightning_model.py --img_size=624 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=592 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=560 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=528 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=496 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=464 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=432 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=400 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=368 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=336 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=304 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=299 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=288 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=256 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=224 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=192 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=160 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=128 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=96 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=64 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=32 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

python3 optuna_lightning_model.py --img_size=24 --real_test --log_name=img_size_trials --n_trials=3 --epochs=15

