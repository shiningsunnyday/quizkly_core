import Vue from 'vue'
import App from './App.vue'
import router from './router'

import Cookies from 'js-cookie';

import "@/assets/css/main.css"


Vue.use(Cookies)

Vue.config.productionTip = false

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')
