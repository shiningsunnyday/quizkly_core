from rest_framework import status
from rest_framework import permissions
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticated, AllowAny
from quizkly_app.models import AppUser, Corpus, Quiz, Question, Distractor
from rest_framework.reverse import reverse
from quizkly_app.serializers import (
    AppUserSerializer,
    UserSerializer,
    CorpusSerializer,
    QuizSerializer,
    QuestionSerializer
)
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from service.question_generator import QuestionGenerator
from service.elmo_client import ElmoClient
from rest_framework.decorators import api_view
import spacy
import os
import json

try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path[:dir_path.index('/web')]
    gen = None
    ec = ElmoClient()
    parser = spacy.load("en_core_web_sm")
    with open(str(os.getenv("MODELS_CONFIG")), 'r') as file:
        json = json.load(file)
        gen = QuestionGenerator(
            json["smp"], json["gmp"], json["wmp"], ec, parser)
except FileNotFoundError:
    pass


class SignUp(APIView):
    parser_classes = (JSONParser,)
    permission_classes = (AllowAny,)

    def post(self, request, format=None):
        username = request.data["username"]
        password = request.data["password"]
        email = request.data["email"]
        user = User.objects.create_user(
            email=email, username=username, password=password)
        appuser = AppUser(user=user)
        user.save()
        appuser.save()
        szuser = UserSerializer(user, data=request.data)
        if szuser.is_valid():
            return Response(szuser.data)
        return HttpResponse(status=status.HTTP_400_BAD_REQUEST)


class Login(APIView):

    parser_classes = (JSONParser,)
    permission_classes = (AllowAny,)

    def post(self, request, format=None):
        username = request.data["username"]
        password = request.data["password"]
        email = request.data["email"]
        user = authenticate(username=username, password=password, email=email)
        if user is None:
            raise AuthenticationFailed("Username/password invalid.")
        else:
            login(request, user)
            sz = UserSerializer(user, data=request.data)
            if sz.is_valid():
                return Response(sz.data)
        return Response(sz.data)


"""because User model doesn't have contact and corpuses itself, must manually
add that to serializers"""


class UserList(generics.ListAPIView):
    permission_classes = (AllowAny,)
    queryset = User.objects.all()
    serializer_class = UserSerializer


class UserDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = (AllowAny,)
    queryset = User.objects.all()
    serializer_class = UserSerializer


def process_corpus(corpus_id, quiz_id, gener=None):
    corpus = Corpus.objects.get(id=corpus_id)
    quiz = Quiz.objects.get(id=quiz_id)
    question_candidates = []
    if gener is None:
        global gen
    else:
        gen = gener
    for batch in gen.generate_questions(corpus.content):
        question_candidates.extend(batch)
    for qc in question_candidates:
        if(len(qc.distractors) >= 2):
            answer = qc.gap.text
            question = qc.question_sentence.replace(answer, "_________")
            ques = Question(quiz=quiz, question=question, correct=0)
            ques.save()
            ans = Distractor(index=0, question=ques, text=answer)
            ans.save()
            for i, dist in enumerate(qc.distractors):
                distractor = Distractor(
                    index=i+1, question=ques, text=dist.text)
                distractor.save()


class CorpusList(APIView):

    permission_classes = (permissions.IsAuthenticated,)

    def get(self, request, format=None):

        corpuses = Corpus.objects.all().filter(user=self.request.user.id)
        sz = CorpusSerializer(corpuses, many=True)
        return Response(sz.data)

    def post(self, request, format=None):

        user = self.request.user
        content = self.request.data["content"]
        appuser = AppUser(user=user)
        corpus = Corpus(user=appuser, content=content)
        corpus.save()
        quiz = Quiz(name=self.request.data["name"], corpus=corpus)
        quiz.save()
        process_corpus(corpus.id, quiz.id)
        sz = CorpusSerializer(corpus)
        return Response(sz.data, status=status.HTTP_201_CREATED)

    def put(self, request, format=None):
        appuser = AppUser(user=self.request.user)
        corpus_id = list(Corpus.objects.all().filter(user=appuser))
        corpus_id = corpus_id[int(self.request.data["quiz_id"])].id
        quiz = Corpus.objects.get(id=corpus_id).quiz
        questions = list(Question.objects.all().filter(quiz=quiz))
        for edit in self.request.data["edits"]:
            if edit[1] < 0:
                question = questions[edit[0]]
                question.question = edit[2]
                question.save()
                continue
            question = questions[edit[1]]
            dists = list(Distractor.objects.all().filter(question=question))
            distractor = dists[edit[0]]
            distractor.text = edit[2]
            distractor.save()
        sz = QuizSerializer(Corpus.objects.get(id=corpus_id).quiz)
        return Response(sz.data)


class CorpusDetail(APIView):

    permission_classes = (permissions.IsAuthenticated,)

    def get_queryset(self):
        queryset = Corpus.objects.all()
        if not self.request.user.is_staff:
            queryset = queryset.filter(owner=self.request.user)
            return queryset

    def delete(self, request, pk, format=None):

        corpus = self.get_object(pk)
        corpus.delete()
        return HttpResponse(status=status.HTTP_204_NO_CONTENT)


class QuizList(generics.ListAPIView):
    permission_classes = (IsAuthenticated,)
    parser_classes = (JSONParser,)

    def get_queryset(self):
        queryset = Quiz.objects.all()
        if('corpus_id' in self.request.data):
            return queryset.filter(corpus=self.request.data["corpus_id"])
        return queryset
    serializer_class = QuizSerializer


class QuizDetail(generics.RetrieveUpdateDestroyAPIView):

    permission_classes = (IsAuthenticated,)
    serializer_class = QuizSerializer

    def put(self, request, pk, format=None):
        appuser = AppUser(user=self.request.user)
        corpus_id = list(Corpus.objects.all().filter(user=appuser))[pk].id
        quiz = Corpus.objects.get(id=corpus_id).quiz
        quiz.name = self.request.data["newTitle"]
        quiz.save()
        sz = QuizSerializer(quiz)
        return Response(sz.data)

    def delete(self, request, pk, format=None):
        quizzes = Quiz.objects.all().filter(
            corpus__user__user=self.request.user)
        quiz = list(quizzes)[pk]
        Distractor.objects.all().filter(question__quiz=quiz).delete()
        Question.objects.all().filter(quiz=quiz).delete()
        quiz.corpus.delete()
        quiz.delete()
        return HttpResponse(status=status.HTTP_204_NO_CONTENT)


class QuestionList(generics.ListAPIView):
    permission_classes = (IsAuthenticated,)
    parser_classes = (JSONParser,)

    def get_queryset(self):
        queryset = Question.objects.all()
        if('ques_id' in self.request.data):
            return queryset.filter(corpus=self.request.data["ques_id"])
        return queryset
    serializer_class = QuestionSerializer


class QuestionDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = (IsAuthenticated,)
    queryset = Question.objects.all()
    serializer_class = QuestionSerializer


class AppUserList(generics.ListCreateAPIView):
    permission_classes = (AllowAny,)
    queryset = AppUser.objects.all()
    serializer_class = AppUserSerializer


class AppUserDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = (IsAuthenticated,)
    queryset = AppUser.objects.all()
    serializer_class = AppUserSerializer


@api_view(['GET'])
def api_root(request, format=None):
    return Response({
        'appusers': reverse('appuser-list', request=request, format=format),
        'users': reverse('user-list', request=request, format=format),
        'quizzes': reverse('quiz-list', request=request, format=format),
        'corpuses': reverse('corpus-list', request=request, format=format),
        'questions': reverse('question-list', request=request, format=format),
        'signup': reverse('signup', request=request, format=format),
        'login': reverse('login', request=request, format=format),
    })
