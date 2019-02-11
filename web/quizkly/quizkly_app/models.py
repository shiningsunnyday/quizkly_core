from django.db import models
from django.contrib.auth.models import User

# for all people who signed up mailing list


class Contact(models.Model):
    submitted = models.DateTimeField(auto_now_add=True)
    email = models.CharField(max_length=30)

    class Meta:
        ordering = ("submitted",)


class Quiz(models.Model):
    submitted = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=30)

    class Meta:
        ordering = ("submitted",)


class Corpus(models.Model):
    submitted = models.DateTimeField(auto_now_add=True)
    quiz = Quiz
    content = models.TextField()

    class Meta:
        ordering = ("submitted",)


class Question(models.Model):
    generated = models.DateTimeField(auto_now_add=True)
    question = models.TextField()

    class Meta:
        ordering = ("generated",)
