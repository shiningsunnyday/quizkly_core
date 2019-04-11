<!-- View for rendering landing page -->
<template>
  <div>
    <NavBar :links="[{path: '#', name: 'About'}, {path: '#', name: 'Contact'}, {path: '/login', name: 'Get Started'}]"/>
    <div class="container">
      <div class="infoContainer">
        <div class="infoText">
          <h1 class="logoText">Quizkly</h1>
          <h1 class="titleText">A <span class="underline">fully automated</span> <br/> quiz generator</h1>
          <h1 class="subjectText">
            All you have to do is copy-paste <br/> 
            your notes and click a button. <br/>
            Quizzes, Quizkly.
          </h1>
        </div>
        <div class="video">
          <iframe width="100%" height="300px" src="https://www.youtube.com/embed/AlQD5QnjfCY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </div>
      </div>
      <div class="mailingForm">
        <input class="emailTextArea" v-model="email" :disabled="this.success"
               placeholder="Email for Mailing List"/>
        <button class="mailingBtn" v-if="!loading" :disabled="this.success"
                v-bind:class="{ disabled: this.success }" 
                v-on:click="addToMailingList">
              {{ getButtonText() }}
        </button>
        <template v-else>
          <div class="lds-ellipsis"><div></div><div></div><div></div><div></div></div>
        </template>
      </div>
    </div>
  </div>
</template>

<script>
import NavBar from '@/components/NavBar.vue'
export default {
  name: 'LandingPage',
  components: {
    NavBar    
  },
  data: function () {
    return {
      // User's email.
      email: "",
      // Whether page is loading.
      loading: false,
      // Whether email submission was successful.
      success: false
    }
  },
  methods: {
    addToMailingList: function () {
      // Submit email to mailing list server.
      const callMailingServer = async () => {
        this.loading = true;
        await fetch('http://shiningsunnyday.pythonanywhere.com/contacts/', {
          method: 'POST',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            email: this.email
          }),
        }).then((response) => {
            this.loading = false;
            if(response.status == 201) {
              this.success = true;
            } else {
              this.$router.push({ name: 'home' });
            }
          }
        ).catch(function() {
          this.$router.push({ name: 'home' });
          }.bind(this)
        )
      }
      callMailingServer();
    },
    getButtonText: function () {
      if(this.success){
        return "Done";
      } else {
        return "Join";
      }
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.container {
  display: flex;
  flex-direction: column;
  margin-right: 3%;
  margin-left: 3%;
  margin-top: -1.5%;
}
.infoContainer {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
}

.infoText {
  flex:1;
  margin-right: 5%;
}

.logoText {
  font-family: Nunito;
  font-style: normal;
  font-weight: normal;
  font-size: 64px;
  line-height: normal;
  color: #E5E5E5;
}

.titleText {
  font-family: Nunito Sans;
  font-style: normal;
  font-weight: normal;
  font-size: 48px;
  line-height: normal;

  color: #32494A;
}

.subjectText {
  font-family: Nunito Sans;
  font-style: normal;
  font-weight: normal;
  font-size: 36px;
  line-height: normal;

  color: #E5E5E5;
}

.video {
  flex: 1;
}

.mailingBtn {
  border: 0;
  padding: 1.5% 80px;
  font-family: Nunito;
  font-style: normal;
  font-weight: 300;
  font-size: 30px;
  background: #3B6167;
  color: #E5E5E5;
  opacity: 0.7;
  border-radius: 10px;
}


.mailingBtn:hover {
  opacity: 0.7;
  box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
}

.disabled {
  opacity: 0.8;
}

.emailTextArea {
  border: 0;
  outline: none;
  padding: 0px 50px;
  background-color: #9CC0BA;
  border-radius: 10px;
  font-family: Nunito;
  font-style: normal;
  font-weight: 300;
  font-size: 30px;
  line-height: normal;
  text-align: center;
  color: #3B6167;
  margin-right:4%;
}

::placeholder {
  color: #E5E5E5;
  opacity: 1; /* Firefox */
}

.mailingForm {
  display: flex;
  flex-direction: row;
  align-self: center;
  margin-top: 5%;
}
</style>
