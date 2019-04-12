<!-- View for rendering signup page -->
<template>
  <div>
    <NavBar :links="[{path: '#', name: 'Quizkly'}]"/>
    <div class="container">
      <h1 class="titleText">Sign Up</h1>
      <div class="loginbox">
        <input class="lightTextBox email" placeholder="Email"
               v-model="email"/>
        <input type="password" class="lightTextBox password" placeholder="Password"
               v-model="password"/>
        <button class="darkBtn" v-if="!loading" v-on:click="login">Sign Up</button>
        <template v-else>
          <div class="lds-ellipsis"><div></div><div></div><div></div><div></div></div>
        </template>
      </div>
      <div v-if="failedSignup" class="errorBox">{{ this.errorMsg }}</div>
    </div>
  </div>
</template>

<script>
import NavBar from '@/components/NavBar.vue'
import Cookies from 'js-cookie';
export default {
  name: 'SignupView',
  components: {
    NavBar
  },
  data: function () {
    return {
      // if signup failed.
      failedSignup: false,
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
      // Calls signup endpoint.
      const callSignupServer = async () => {
        this.loading = true;
        await fetch(
          'http://localhost:8000/signup/', {
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
              this.failedSignup = false;
              this.$router.push({ name: 'login' });
            } else {
              this.failedSignup = true;
              this.errorMsg = "Signup failed. Try again.";
            }
          }.bind(this)
        ).catch(function() {
            this.loading = false;
            this.failedSignup = true;
            this.errorMsg = "Server is down. Try again later.";
          }.bind(this)
        );
      }
      callSignupServer();
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped src="@/assets/css/login.css"> </style>
