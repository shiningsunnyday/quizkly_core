from rest_framework import serializers
from quizkly_app.models import Contact, Corpus, Quiz, Question
from django.contrib.auth.models import User


class ContactSerializer(serializers.HyperlinkedModelSerializer):
    email = serializers.CharField(max_length=30)

    class Meta:
        model = Contact
        fields = ("url", "user_id", "email")


class UserSerializer(serializers.HyperlinkedModelSerializer):
    corpuses = serializers.HyperlinkedRelatedField(
        many=True,
        lookup_field="owner",
        view_name="corpus-detail",
        queryset=Corpus.objects.all(),
        allow_null=True,
    )
    contact = serializers.HyperlinkedRelatedField(
        lookup_field="user_id",
        view_name="contact-detail",
        allow_null=True,
        queryset=Contact.objects.all(),
    )

    class Meta:
        model = User
        fields = ("url", "id", "username", "password", "contact", "corpuses")


class CorpusSerializer(serializers.HyperlinkedModelSerializer):
    content = serializers.CharField(style={"base_template": "textarea.html"})
    owner = serializers.ReadOnlyField(source="owner.username")
    quiz = serializers.HyperlinkedRelatedField(
        view_name="quiz-detail",
        lookup_field="corpus_id",
        queryset=Quiz.objects.all(),
        allow_null=True,
    )

    class Meta:
        model = Corpus
        fields = ("url", "id", "owner", "content", "quiz")


class QuizSerializer(serializers.HyperlinkedModelSerializer):
    questions = serializers.HyperlinkedRelatedField(
        many=True,
        view_name="question-detail",
        lookup_field="quiz_id",
        queryset=Question.objects.all(),
    )
    corpus_id = serializers.ReadOnlyField(source="corpus_id.title")

    class Meta:
        model = Quiz
        fields = ("url", "id", "corpus_id", "questions")


class QuestionSerializer(serializers.HyperlinkedModelSerializer):
    question = serializers.CharField(style={"base_template": "textarea.html"})
    quiz_id = serializers.ReadOnlyField(source="quiz_id.title")

    class Meta:
        model = Question
        fields = ("url", "quiz_id", "question")
