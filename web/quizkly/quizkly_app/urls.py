from django.urls import path
from django.conf.urls import include
from quizkly_app import views
from django.views.decorators.csrf import csrf_exempt
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('', views.api_root),
    path("signup/", views.SignUp.as_view(), name="signup"),
    path("login/", views.Login.as_view(), name="login"),
    path("appusers/", views.AppUserList.as_view(), name="appuser-list"),
    path("appusers/<int:pk>/", views.AppUserDetail.as_view(), name="appuser-detail"),
    path("users/", views.UserList.as_view(), name="user-list"),
    path("users/<int:pk>/", views.UserDetail.as_view(), name="user-detail"),
    path("quizzes/", views.QuizList.as_view(), name="quiz-list"),
    path("quizzes/<int:pk>/", views.QuizDetail.as_view(), name="quiz-detail"),
    path("corpuses/", views.CorpusList.as_view(), name="corpus-list"),
    path("corpuses/<int:pk>/", views.CorpusDetail.as_view(), name="corpus-detail"),
    path("questions/<int:pk>/", views.QuestionDetail.as_view(), name="question-detail"),
    path("questions/", views.QuestionList.as_view(), name="question-list"),
]

urlpatterns += [path("api-auth/", include("rest_framework.urls"))]

urlpatterns = format_suffix_patterns(urlpatterns)
