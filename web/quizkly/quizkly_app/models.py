from django.db import models
from django.contrib.auth.models import User

# for all people who signed up mailing list

class AppUser(models.Model):
    registered = models.DateTimeField(auto_now_add = True)
    user = models.OneToOneField(User, on_delete = models.CASCADE, primary_key = True)
    class Meta:
        ordering = ("registered",)

class Corpus(models.Model):
    user = models.ForeignKey(AppUser, on_delete = models.CASCADE)
    submitted = models.DateTimeField(auto_now_add=True)
    content = models.TextField()
    class Meta:
        ordering = ("submitted",)

class Quiz(models.Model):
    corpus = models.OneToOneField(Corpus, on_delete = models.CASCADE)
    generated = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=30)
    class Meta:
        ordering = ("generated",)

class Question(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete = models.CASCADE)
    generated = models.DateTimeField(auto_now_add=True)
    question = models.TextField()
    correct = models.IntegerField()
    class Meta:
        ordering = ("generated",)

class Distractor(models.Model):
    index = models.IntegerField()
    generated = models.DateTimeField(auto_now_add=True)
    question = models.ForeignKey(Question, on_delete = models.CASCADE)
    text = models.CharField(max_length=30)
    class Meta:
        ordering = ("generated",)
