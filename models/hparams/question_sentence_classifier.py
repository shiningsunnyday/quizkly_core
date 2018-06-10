from models.sentence_classifier import HParams

baseline = HParams(
    train_records="local/squad/baltrain.tfrecords",
    eval_records="local/squad/baltest.tfrecords",
    sentence_feature="sentence",
    label_feature="question_worthy",
    train_batch_size=100,
    eval_batch_size=100,
    hidden_size=512,
    num_classes=2
)