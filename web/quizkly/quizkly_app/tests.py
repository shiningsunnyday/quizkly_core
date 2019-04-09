from django.test import TestCase
from quizkly_app.models import User, AppUser, Quiz, Corpus
from django.contrib.auth import authenticate
import spacy
from service.question_generator import QuestionGenerator
from quizkly_app.views import process_corpus
from datasets.create_test_data import DummyElmoClient

gen = None

gap_model_path = "../../models/test_data/gap_classifier.h5"
sentence_model_path = "../../models/test_data/sentence_classifier/saved_model"
word_model_path = "../../models/test_data/word_model"
parser = spacy.load("en_core_web_sm")
gen = QuestionGenerator(sentence_model_path, gap_model_path, word_model_path,
                        DummyElmoClient(), parser)

user_data = {
    "email": "girishkumar@gmail.com",
    "username": "girishkumar",
    "password": "girishisku",
}

test_data = {
    "content": '''
        In molecular biology, DNA replication is the biological process of
        producing two identical replicas of DNA from one original DNA molecule.
        DNA replication occurs in all living organisms acting as the basis for
        biological inheritance. The cell possesses the distinctive property of
        division, which makes replication of DNA essential. DNA is made up of a
        double helix of two complementary strands. During replication, these
        strands are separated. Each strand of the original DNA molecule then
        serves as a template for the production of its counterpart, a process
        referred to as semiconservative replication. As a result of
        semi-conservative replication, the new helix will be composed of an
        original DNA strand as well as a newly synthesized strand.[1] Cellular
        proofreading and error-checking mechanisms ensure near perfect fidelity
        for DNA replication.[2][3] In a cell, DNA replication begins at
        specific locations, or origins of replication, in the genome.[4]
        Unwinding of DNA at the origin and synthesis of new strands,
        accommodated by an enzyme known as helicase, results in replication
        forks growing bi-directionally from the origin. A number of proteins
        are associated with the replication fork to help in the initiation and
        continuation of DNA synthesis. Most prominently, DNA polymerase
        synthesizes the new strands by adding nucleotides that complement each
        (template) strand. DNA replication occurs during the S-stage of
        interphase. DNA replication (DNA amplification) can also be performed
        in vitro (artificially, outside a cell). DNA polymerases isolated from
        cells and artificial DNA primers can be used to start DNA synthesis at
        known sequences in a template DNA molecule. Polymerase chain reaction
        (PCR), ligase chain reaction (LCR), and transcription-mediated
        amplification (TMA) are examples.
    ''',
    "quiz_name": "DNA Replication"
}


class QuizTestCase(TestCase):

    def setUp(self):
        test_user = User.objects.create_user(
            email=user_data["email"],
            username=user_data["username"],
            password=user_data["password"]
        )
        test_app_user = AppUser.objects.create(user=test_user)
        self.assertNotEqual(test_app_user, None)
        test_user = authenticate(
            email=user_data["email"],
            username=user_data["username"],
            password=user_data["password"]
        )

    def test_user(self):
        test_user = User.objects.get(email=user_data["email"])
        self.assertNotEqual(test_user, None)
        self.assertEqual(test_user.username, user_data["username"])
        test_app_user = AppUser.objects.get(user=test_user)
        self.assertNotEqual(test_app_user, None)

    def test_quiz_create(self):
        test_app_user = AppUser.objects.get(user__email=user_data["email"])
        corpus = Corpus.objects.create(user=test_app_user)
        quiz = Quiz.objects.create(corpus=corpus, name=test_data['quiz_name'])
        global gen
        process_corpus(corpus.id, quiz.id, gen)
