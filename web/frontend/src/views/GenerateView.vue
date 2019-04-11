<!-- View for rendering new quiz generation page -->
<template>
  <div>
    <NavBar :links="[{path: '/', name: 'Quizkly'}]"/>
    <div class="container">
      <h1 class="titleText">New Quiz</h1>
      <input class="textInput nameBox" v-bind:class="{disabled: loading}"
             placeholder="Quiz name" v-model="quizTitle" :disabled="loading"/>
      <textarea class="textInput copyArea" v-bind:class="{disabled: loading}"
                placeholder="Your Text" v-model="genText" :disabled="loading">          
      </textarea>
      <button v-if="!loading" class="darkBtn" v-on:click="generateQuiz">
        Generate
      </button>
      <template v-else>
        <div class="lds-ellipsis"><div></div><div></div><div></div><div></div></div>
      </template>
    </div>
  </div>
</template>

<script>
import NavBar from '@/components/NavBar.vue'
import Cookies from 'js-cookie';
export default {
  name: 'GenerateView',
  components: {
    NavBar 
  },
  data: function () {
    return {
      // Title of quiz.
      quizTitle: "",
      // Text to generate quizzes from.
      genText: "",
      // Whether the page is loading
      loading: false
    }
  },
  methods: {
    generateQuiz: function() {
      const callCorpusServer = async () => {
        // Server generates quiz and user is routed to quiz page.
        this.loading = true;
        let response = await fetch('http://localhost:8000/corpuses/', {
            credentials: 'include',
            method: 'POST',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json',
              'X-CSRFToken': Cookies.get('csrftoken'),
            },
            body: JSON.stringify({
              name: this.quizTitle,
              content: this.genText,
            }),
          })
        let corpus = await response.json();
        this.loading = false;
        this.$router.push(
          { name: 'quiz',
            params: { title: corpus.quiz.name, 
                      questions: corpus.quiz.question_set
            } 
          }
        )
      }
      callCorpusServer();
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
  margin-top: -5%;
}

.textInput {
  border: 0;
  outline: none;
  background-color: #9CC0BA;
  border-radius: 10px;
  font-family: Nunito;
  font-style: normal;
  font-weight: 300;
  font-size: 15px;
  line-height: normal;
  text-align: left;
  color: #3B6167;
  padding: 10px;
  width: 90%;
  align-self: center;
}

.copyArea {
  font-size: 15px;
  width: 90%;
  height: 450px;
}

.nameBox {
  font-size: 28px;
  width: 90%;
  margin-bottom: 1%;
}

::placeholder {
  color: #E5E5E5;
  font-size: 28px;
  opacity: 1; /* Firefox */
}

.darkBtn {
  border: 0;
  font-family: Nunito;
  font-style: normal;
  font-weight: 300;
  font-size: 28px;
  background: #3B6167;
  color: #E5E5E5;
  opacity: 0.7;
  border-radius: 10px;
  margin-bottom: 20px;
  padding: 5px 50px; 
  margin-top: 20px;
  align-self: center;
}

.darkBtn:hover {
  opacity: 0.7;
  box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
}

.disabled {
  opacity: 0.5;
}
</style>
