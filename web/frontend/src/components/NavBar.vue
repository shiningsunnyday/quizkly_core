<!-- Navigation Bar Component -->
<template>
  <ul class="navigation">
    <router-link v-for="link in links" tag="li" :to="`${link.path}`" :key="link.path">
      <a>{{link.name}}</a>
    </router-link>
    <a v-if="loggedIn" v-on:click.prevent="logout">Logout</a>
  </ul>
</template>

<script>
export default {
  name: 'NavBar',
  props: {
    /** 
     * Array of {link path, rendered name} tuples to be
     * shown in navigation bar.
     */
    links: Array
  },
  data: function () {
    return {
      // whether user is logged in
      loggedIn: localStorage.getItem('jwt') != null,
    }
  },
  methods: {
    logout: function () {
      /**
       * Logs out by removing auth token for local storage
       * and rerouting to homepage.
       */
      localStorage.removeItem('jwt');
      this.loggedIn = false;
      this.$forceUpdate();
      this.$router.push({ name: 'home' });
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.navigation {
  list-style: none;
  margin: 0;
  
  display: -webkit-box;
  display: -moz-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  
  -webkit-flex-flow: row wrap;
  justify-content: flex-end;
}

.navigation a {
  text-decoration: none;
  display: block;
  padding: 1em;
  color: #E5E5E5;
}

.navigation a:hover {
  opacity: 0.8;
  -webkit-transition-duration: 0.4s; /* Safari */
  transition-duration: 0.4s;
  cursor: pointer;
}
</style>
