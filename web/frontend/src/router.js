import Vue from 'vue'
import Router from 'vue-router'
import LandingPage from './views/LandingPage.vue'
import GenerateView from './views/GenerateView.vue'
import QuizView from './views/QuizView.vue'
import LoginView from './views/LoginView'
import SignupView from './views/SignupView'

Vue.use(Router)

let router = new Router({
  routes: [
    {
      path: '/',
      name: 'home',
      component: LandingPage
    },
    {
      path: '/login',
      name: 'login',
      component: LoginView,
      meta: {
        guest: true
      }
    },
    {
      path: '/signup',
      name: 'signup',
      component: SignupView,
      meta: {
        guest: true
      }
    },
    {
      path: '/quiz',
      name: 'quiz',
      props: true,
      component: QuizView,
      meta: {
        requiresAuth: true
      }
    },
    {
      path: '/generate',
      name: 'generate',
      component: GenerateView,
      meta: {
        requiresAuth: true
      }
    },
  ]
})

// Check authorization before routing to protected views.
router.beforeEach((to, from, next) => {
  if(to.matched.some(record => record.meta.requiresAuth)) {
      if (localStorage.getItem('jwt') == null) {
        next({path: 'login'});
      } else {
        next();
      }
  } else if(to.matched.some(record => record.meta.guest)) {
      if(localStorage.getItem('jwt') == null){
        next();
      }
      else{
        next({ name: 'generate'});
      }
  } else {
    next();
  }
})

export default router;
