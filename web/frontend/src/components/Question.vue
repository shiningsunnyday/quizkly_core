<!-- Component that renders a single question -->
<template>
  <div class="questionContainer">
    <div class="question">
      {{ questionObj.question }}
    </div>  
    <div class="distractors">
      <button v-for="set in distTextIds" :id="`${set.id}`" :class="getClass(set)" :key="set.id" v-on:click="updateAnswer($event)">
        <svg v-if="showTick(set)"
            class="btnIcon" width="24" height="24"
            xmlns="http://www.w3.org/2000/svg" fill-rule="evenodd"
            clip-rule="evenodd">
            <path d="M21 6.285l-11.16 12.733-6.84-6.018 1.319-1.49 5.341 4.686 9.865-11.196 1.475 1.285z"/>
        </svg>
        {{set.text}}
      </button>
    </div>        
  </div>
</template>

<script>
export default {
  name: 'Question',
  props: {
    /**
     * Object that contains the properties 
     * `question` which contains the question text
     * `distractor_set` which contains the property text i.e. dist.text
     * `correct` which contains the index of the correct distractor as in `distractor_set`
     */
    questionObj: Object
  },
  computed: {
    distTextIds: function () {
      /**
       * Converts distractor set into an array of id, string tuples and randomizes order.
       */
      var arr = [];
      for (let [index, dist] of this.questionObj.distractor_set.entries()) {  
        arr.push({id: index, text: dist.text});
      }
      arr.sort(() => Math.random() - 0.5);
      return arr
    },
    correctId: function() {
      // Returns index of correct question.
      return this.questionObj.correct
    },
    answered: function() {
      // Returns if questions has been answered.
      return this.clickedId != -1
    },
    correct: function() {
      // Returns if answer is correct.
      return this.answered && this.clickedId == this.correctId
    },
    wrong: function() {
      // Returns if answer is wrong.
      return this.answered && this.clickedId != this.correctId
    }
  },
  data: function() {
    return {
      // Tracks id of distractor that's been clicked.
      clickedId: -1,
    }
  },
  methods: {
    updateAnswer: function(event) {
      // Update selected answer.
      this.clickedId = event.currentTarget.id;
    },
    getClass: function(set) {
      // Choose class of each button.
      return {
            'distBtn': true,  
            'answered': this.answered,
            'showCorrect': this.answered && set.id == this.correctId,
            'wronglyAnswered': this.wrong && this.clickedId == set.id,
      }
    },
    showTick: function(set) {
      // Whether to show tickmark to indicate a correct answer.
      return this.answered && set.id == this.correctId;
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.question {
  background: #304D58;
  padding: 1% 1%;
  border-radius: 10px 10px 0px 0px;
  font-family: Nunito;
  font-style: normal;
  font-weight: 200;
  font-size: 22px;
  line-height: normal;
  text-align: center;
  color: #E5E5E5;
}

.distractors {
  background: #3C6268;
  padding-top: 1%;
  padding-bottom: 1%;
  border-radius: 0px 0px 10px 10px;
  display: flex;
  flex-flow: row wrap;
  justify-content: space-around;
}

.distBtn {
  background: #C4C4C4;
  border-radius: 5px;
  font-family: Nunito;
  font-style: normal;
  font-weight: 200;
  font-size: 20px;
  line-height: normal;
  text-align: center;
  color: #304D58;
  min-width: 225px;
  width: 19%;
  padding: 0.5% 0.5%;
  margin-bottom: 2px;
  margin-top: 2px;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
}

.answered {
  opacity: 0.6;
  pointer-events: none;
}

.showCorrect {
  background: #304D58;;
  box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
  color: #C4C4C4;
  opacity: 1.0;
}

.wronglyAnswered {
  background: #9F4A35;
  box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
  color: #C4C4C4;
  opacity: 1.0;
}

.distBtn:hover {
  box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
  background: #304D58;
  color: #C4C4C4;
}

.btnIcon {
  align-self: flex-start;
  fill: #fff;
}
</style>
