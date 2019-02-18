from rest_framework import status
from rest_framework import permissions
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.exceptions import ParseError, AuthenticationFailed
from rest_framework.parsers import JSONParser
from rest_framework.renderers import JSONRenderer
from rest_framework.permissions import IsAuthenticated, AllowAny
from quizkly_app.models import AppUser, Corpus, Quiz, Question, Distractor
from rest_framework.reverse import reverse
from quizkly_app.serializers import (
    AppUserSerializer,
    UserSerializer,
    CorpusSerializer,
    QuizSerializer,
    QuestionSerializer,
    DistractorSerializer
)
from django.http import Http404
from rest_framework.metadata import SimpleMetadata
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from quizkly_app.permissions import IsOwnerOrReadOnly, Always
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.utils.decorators import method_decorator
from django.template import RequestContext, Template
from service.question_generator import QuestionGenerator
from service.elmo_client import ElmoClient

gen = None

class SignUp(APIView):
    parser_classes = (JSONParser,)
    permission_classes = (AllowAny,)

    def post(self, request, format=None):
        username = request.data["username"]
        password = request.data["password"]
        email = request.data["email"]
        user = User.objects.create_user(email = email, username=username, password=password)
        appuser = AppUser(user = user)
        user.save()
        appuser.save()
        szuser = UserSerializer(user, data = request.data)
        szappuser = AppUserSerializer(appuser, data = request.data)
        if szuser.is_valid() and szappuser.is_valid():
            szuser.save()
            szappuser.save()
        print(szuser.data)
        print(szappuser.data)
        return Response(szuser.data)


class Login(APIView):

    parser_classes = (JSONParser,)
    permission_classes = (AllowAny,)

    def post(self, request, format=None):
        print(request)
        username = request.data["username"]
        password = request.data["password"]
        email = request.data["email"]
        user = authenticate(username=username, password=password, email=email)
        if user is None:
            raise AuthenticationFailed("Username/password invalid.")
        else:
            login(request, user)
            print(request.data)
            sz = UserSerializer(user, data=request.data)
            if sz.is_valid():
                print("Valid")
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


def process_corpus(corpus_id, quiz_id):
    print("Let's process")
    corpus = Corpus.objects.get(id = corpus_id)
    quiz = Quiz.objects.get(id = quiz_id)
    print(corpus.content, " is content")
    smp = "/Users/shiningsunnyday/Documents/GitHub/kvizo_core/models/1547524090"
    gmp = "/Users/shiningsunnyday/Documents/GitHub/kvizo_core/models/gap_model"
    wmp = "/Users/shiningsunnyday/Documents/GitHub/kvizo_core/models/wmdatabio70.bin"
    ec = ElmoClient()
    global gen
    if gen == None:
        print("need new gen")
        gen = QuestionGenerator(smp, gmp, wmp, ec)
    question_candidates = []
    for batch in gen.generate_questions(corpus.content):
        question_candidates.extend(batch)
    for qc in question_candidates:
        answer = qc.gap.text
        question = qc.question_sentence.replace(answer, "_________")
        print("Question", question)
        ques = Question(quiz = quiz, question = question, correct = 0)
        ques.save()
        print(ques)
        ans = Distractor(index = 0, question = ques, text = answer)
        ans.save()
        print("Answer", ans)
        for i, dist in enumerate(qc.distractors):
            print(i, dist.text)
            distractor = Distractor(index = i + 1, question = ques, text = dist.text)
            distractor.save()
            print("Distractor", distractor)
        print("Answer", answer)

# process_corpus(sz.data['id'])
class CorpusList(APIView):

    permission_classes = (permissions.IsAuthenticated,)

    def get(self, request, format=None):

        corpuses = Corpus.objects.all().filter(user = self.request.user.id)
        sz = CorpusSerializer(corpuses, many=True)
        return Response(sz.data)

    def post(self, request, format=None):

        user = self.request.user
        content = self.request.data["content"]
        print(self.request.data)
        appuser = AppUser(user = user)
        corpus = Corpus(user = appuser, content = content)
        corpus.save()
        quiz = Quiz(name = self.request.data["name"], corpus = corpus)
        quiz.save()
        process_corpus(corpus.id, quiz.id)
        print(data)
        sz = CorpusSerializer(corpus, data=data)
        print(repr(sz))
        if sz.is_valid():
            sz.save(user = appuser)
            return Response(sz.data, status=status.HTTP_201_CREATED)
        return Response(sz.errors, status=status.HTTP_400_BAD_REQUEST)


class CorpusDetail(APIView):

    permission_classes = (permissions.IsAuthenticated,)

    def get_queryset(self):
        queryset = Corpus.objects.all()
        if not self.request.user.is_staff:
            queryset = queryset.filter(owner=self.request.user)
            return queryset

    def get_object(self, pk):
        try:
            corpus = Corpus.objects.get(pk=pk)
        except Corpus.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):

        corpus = self.get_object(pk)
        serializer = CorpusSerializer(corpus, context={"request": request})
        return Response(serializer.data)

    def put(self, request, pk, format=None):

        corpus = self.get_object(pk)
        serializer = Corpus(contact, data=request.data, context={"request": request})
        if serializer.is_valid():
            serializer.save()
            process_corpus(serializer.data["id"])
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

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
            return queryset.filter(corpus = self.request.data["corpus_id"])
        return queryset
    serializer_class = QuizSerializer


class QuizDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = (IsAuthenticated,)
    queryset = Quiz.objects.all()
    serializer_class = QuizSerializer


class QuestionList(generics.ListAPIView):
    permission_classes = (IsAuthenticated,)
    parser_classes = (JSONParser,)
    def get_queryset(self):
        queryset = Question.objects.all()
        if('ques_id' in self.request.data):
            return queryset.filter(corpus = self.request.data["ques_id"])
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

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse

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
