from models.sentence_classifier import HParams

baseline = HParams(
    train_records="local/squad/squad_context_train/contrain.tfrecords",
    eval_records="local/squad/squad_context_train/contest.tfrecords",
    sentence_feature="sentence",
    context_feature="context",
    label_feature="question_worthy",
    train_batch_size=100,
    eval_batch_size=100,
    hidden_size=512,
    num_classes=2,
    learning_rate=0.0005,
)
