<!-- View for rendering login page -->
<template>
  <div>
    <NavBar :links="[{path: '#', name: 'Quizkly'}]"/>
    <div class="container">
      <h1 class="titleText">Login</h1>
      <div class="loginbox">
        <input class="lightTextBox email" placeholder="Email"
               v-model="email"/>
        <input type="password" class="lightTextBox password" placeholder="Password" v-model="password"/>
        <button class="darkBtn" v-if="!loading" v-on:click="login">Login</button>
        <template v-else>
          <div class="lds-ellipsis"><div></div><div></div><div></div><div></div></div>
        </template>
        <div class="signupText">
            Not Registered? 
          <router-link to="/signup" class="signupText">
            <span class="underline">Create an account.</span>
          </router-link>
        </div>
      </div>
      <div v-if="failedLogin" class="errorBox">{{ this.errorMsg }}</div>
    </div>
  </div>
</template>

<script>
import NavBar from '@/components/NavBar.vue'
import Cookies from 'js-cookie';
export default {
  name: 'LoginView',
  components: {
    NavBar
  },
  data: function () {
    return {
      // if login failed.
      failedLogin: false,
      // if page is loading.
      loading: false,
      // user email.
      email: "",
      // user password.
      password: "",
      // message to display, if an error occurs.
      errorMsg: ""  
    }
  },
  methods: {
    login: function () {
      // Calls login endpoint.
      const callLoginServer = async () => {
        this.loading = true;
        await fetch(
          'http://localhost:8000/login/', {
            credentials: 'include',
            method: 'POST',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json',
              'X-CSRFToken': Cookies.get('csrftoken'),
            },
            body: JSON.stringify({
              email: this.email,
              username: this.email,
              password: this.password,
            }),
          }
        ).then(function(response) {
            this.loading = false;
            if (response.ok) {
              this.failedLogin = false;
              localStorage.setItem('jwt', 'auth');
              this.$router.push({ name: 'generate' });
            } else {
              this.failedLogin = true;
              this.errorMsg = "Login failed. Try again.";
            }
          }.bind(this)
        ).catch(function() {
            this.loading = false;
            this.failedLogin = true;
            this.errorMsg = "Server is down. Try again later.";
          }.bind(this)
        );
      }
      callLoginServer();
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped src="@/assets/css/login.css"> </style>
