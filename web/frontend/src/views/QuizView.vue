<!-- View to render a quiz -->
<template>
  <div>
    <NavBar :links="[{path: '/', name: 'Quizkly'}]"/>
    <div class="container">
      <h1 class="titleText">{{title}}</h1>
      <div class="quizContainer">
        <div class="optionsBar">
          <button class="optionsBarBtn">Share</button>
          <button class="optionsBarBtn">Edit</button>
          <router-link tag="button" class="optionsBarBtn" to="/generate">Back</router-link>
        </div>
        <div class="questionsList">
          <Question v-for="question in questions" v-bind:question-obj="question" :key="question.text"></Question>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import NavBar from '@/components/NavBar.vue'
import Question from '@/components/Question.vue'
export default {
  name: 'QuizView',
  props: {
    // Title of Quiz
    title: String,
    // List of question objects to render.
    questions: Array
  },
  components: {
    NavBar, Question 
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

.quizContainer {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
}

.optionsBar {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  align-self: flex-start;
  position: -webkit-sticky;
  position: sticky;
  top: 10%;
  margin-right: 20px;
}

.optionsBarBtn {
  font-family: Nunito;
  font-style: normal;
  font-weight: 300;
  font-size:calc(12px + 1.5vw);
  line-height: normal;
  color: #3C6268;
  background: transparent;
  border-width: 0 0 0 4px;
  border-style: solid;
  border-color: #2B2732;
  text-align: left;
  margin-bottom: 2%;

}

.optionsBarBtn:hover {
  color: #e5e5e5;
  opacity: 0.5;
}

.questionsList {
  flex-grow: 4;
  display: flex;
  flex-direction: column;
}

.questionContainer {
  display: flex;
  flex-direction: column;
  margin-bottom: 1%;
}
</style>
