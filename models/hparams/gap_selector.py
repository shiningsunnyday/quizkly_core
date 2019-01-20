from models.pointer_network import HParams

hparams = HParams(
    train_records="local/squad/len/train*.tfrecords",
    eval_records="local/squad/len/test*.tfrecords",
    sentence_feature="sentence",
    start_label_feature="answer_start",
    end_label_feature="answer_end",
    sentence_length_feature="sentence_length",
    train_batch_size=50,
    eval_batch_size=50,
    hidden_size=100,
    dropout_keep_prob=0.2,
    learning_rate=0.001,
)
