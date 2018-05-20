To train models:

python3 -m trainer.run_training \
--model_class models.mock_model.MockModel \
--hparams_class models.mock_model.HPARAMS \
--model_dir /tmp/mock