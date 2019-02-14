from rest_framework import serializers
from quizkly_app.models import AppUser, Corpus, Quiz, Question, Distractor
from django.contrib.auth.models import User

class DistractorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Distractor
        fields = ("id", "text")

class QuestionSerializer(serializers.ModelSerializer):
    distractor_set = DistractorSerializer(many = True)
    class Meta:
        model = Question
        fields = ("id", "question", "distractor_set", "correct")

class QuizSerializer(serializers.ModelSerializer):
    question_set = QuestionSerializer(many = True)
    class Meta:
        model = Quiz
        fields = ("id", "corpus", "name", "question_set")

class CorpusSerializer(serializers.ModelSerializer):
    content = serializers.CharField()
    quiz = QuizSerializer()
    class Meta:
        model = Corpus
        fields = ("id", "user", "content", "quiz")

class AppUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = AppUser
        fields = ('user',)

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'password', 'email')
