from rest_framework import status
from rest_framework import permissions
from rest_framework import generics
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.exceptions import ParseError, AuthenticationFailed
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticated, AllowAny
from quizkly_app.models import Contact, Corpus, Quiz, Question
from quizkly_app.serializers import (
    ContactSerializer,
    UserSerializer,
    CorpusSerializer,
    QuizSerializer,
    QuestionSerializer,
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


class SignUp(APIView):
    parser_classes = (JSONParser,)
    permission_classes = (AllowAny,)

    def post(self, request, format=None):
        username = request.data["username"]
        password = request.data["password"]
        user = User.objects.create_user(username=username, password=password)
        user.save()
        sz = UserSerializer(data=request.data, context={"request": request})
        sz.corpuses = None
        sz.contact = request.data["contact"]
        if sz.is_valid():
            sz.save()
        return Response(sz.data)


class Login(APIView):

    parser_classes = (JSONParser,)
    permission_classes = (AllowAny,)

    def post(self, request, format=None):
        username = request.data["username"]
        password = request.data["password"]
        user = authenticate(username=username, password=password)
        if user is None:
            raise AuthenticationFailed("Username/password invalid.")
        else:
            login(request, user)
            sz = UserSerializer(data=request.data, context={"request": request})
            sz.corpuses = None
            print(repr(sz))
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


def process_corpus(corpus_id):
    print("Let's process")
    corpus = Corpus.objects.get(id=corpus_id)

    print(corpus.content, " is content")
    smp = "/Users/shiningsunnyday/Documents/GitHub/kvizo_core/models/1547524090"
    gmp = "/Users/shiningsunnyday/Documents/GitHub/kvizo_core/models/gap_model"
    wmp = (
        "/Users/shiningsunnyday/Documents/GitHub/kvizo_core/models/test_data/word_model"
    )
    ec = ElmoClient()
    gen = QuestionGenerator(smp, gmp, wmp, ec)
    question_candidates = []
    for batch in gen.generate_questions(corpus.content):
        question_candidates.extend(batch)
    for qc in question_candidates:
        answer = qc.gap.text
        question = qc.question_sentence.replace(answer, "_________")
        distractors = "\n".join(
            "%d. %s" % (i, dist.text) for i, dist in enumerate(qc.distractors)
        )
        print("Answer ", answer)
        print("Question ", question)
        print("Distractors ", distractors)


# process_corpus(sz.data['id'])
class CorpusList(APIView):

    permission_classes = (permissions.IsAuthenticated,)

    def get(self, request, format=None):

        corpuses = Corpus.objects.all()
        sz = CorpusSerializer(corpuses, many=True)
        return Response(sz.data)

    def post(self, request, format=None):

        sz = CorpusSerializer(data=request.data, context={"request": request})
        sz.quiz = None
        if sz.is_valid():
            sz.save(owner=self.request.user)
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
    queryset = Quiz.objects.all()
    serializer_class = QuizSerializer


class QuizDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = (IsAuthenticated,)
    queryset = Quiz.objects.all()
    serializer_class = QuizSerializer


class QuestionList(generics.ListAPIView):
    permission_classes = (IsAuthenticated,)
    queryset = Question.objects.all()
    serializer_class = QuestionSerializer


class QuestionDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = (IsAuthenticated,)
    queryset = Question.objects.all()
    serializer_class = QuestionSerializer


class ContactList(generics.ListAPIView):
    permission_classes = (IsAuthenticated,)
    queryset = Contact.objects.all()
    serializer_class = ContactSerializer


class ContactDetail(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = (IsAuthenticated,)
    queryset = Contact.objects.all()
    serializer_class = ContactSerializer
