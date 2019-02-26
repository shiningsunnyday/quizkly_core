""" HParams for ranking distractors. """
from models.distractor_encoder import HParams

baseline = HParams(
    train_records="local/triviaqa_records/train*.tfrecords",
    eval_records="local/triviaqa_records/test*.tfrecords",
    question_feature="question",
    answer_feature="answer",
    distractor_feature="distractor",
    negatives_feature="negatives",
    negatives_length_feature="negatives_length",
    vector_size=70,
    train_batch_size=50,
    eval_batch_size=50,
    learning_rate=0.001,
)
