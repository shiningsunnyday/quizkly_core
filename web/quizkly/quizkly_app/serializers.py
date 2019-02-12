from rest_framework import serializers
from quizkly_app.models import AppUser, Corpus, Quiz, Question, Distractor
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'password', 'email')

class AppUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = AppUser
        fields = ('user',)

class CorpusSerializer(serializers.ModelSerializer):
    class Meta:
        model = Corpus
        fields = ("id", "user", "content")

class QuizSerializer(serializers.ModelSerializer):
    class Meta:
        model = Quiz
        fields = ("id", "corpus", "name")

class QuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Question
        fields = ("id", "quiz", "question", "correct")

class DistractorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Distractor
        fields = ("id", "index", "question")
