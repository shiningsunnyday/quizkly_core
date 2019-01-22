import React, { Component } from 'react';
import './App.css';
import { Link } from 'react-router-dom';
import { Container1, Container2, BeforeAfterFlow, NavBar } from './Containers/Container.js';
import './style.css';
import './Components/Components.css';

import Header from './Components/Horizontal/Header.js';
import Mailing from './Components/Horizontal/Mailing.js';
import Title from './Components/Horizontal/Title.js';
import Video from './Components/Horizontal/Video.js';
import Visual from './Components/Horizontal/Visual.js';

import Interface from './Components/Horizontal/Interface.js';


class Quizkly extends Component {

  componentDidMount() {
    this.updateWindowDimensions();

    var whoWeAre = document.getElementById("whoWeAre");
    whoWeAre.style.cursor = 'pointer';
    whoWeAre.onclick = function() {
      this.setState({about: !this.state.about});
    }.bind(this)

    window.addEventListener('resize', this.updateWindowDimensions);
  }

  updateWindowDimensions() {
    this.setState({ width: window.innerWidth, height: window.innerHeight }, console.log(this.state.width, this.state.height));
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.updateWindowDimensions);
  }

  constructor(props) {
    super(props);
    this.state = { width: 0, height: 0 };
    this.updateWindowDimensions = this.updateWindowDimensions.bind(this);
  }

  state = {
    about: false,
    hero: "hero__mask",
    width: 0,
    height: 0,
    demoValue: "",
    navBar: {
      buttonTitles: [
        { title: 'Request a demo' },
        { title: 'Sign up for mailing list' },
        { title: 'About us' }
      ],
      show: false,
      target: null,
      container: this,
      pop: null,
      popName: "",
    },
    beforeAfterElements: [
      { before: 'My learning effiency is low. My memorization ability isn\'t good, so I have to spend a lot of time creating flashcards to test myself with.',
        after: 'I can automate my entire routine with Quizkly! All I need is paste in the content I have to memorize and out pops multiple choice quizzes on demand.' },
      { before: 'Testing myself is like playing cards with myself, and I don\'t have anyone who\'s always there to test my learning. Even when studying with a friend, my friend doesn\'t know my level of understanding and I may easily get distracted.',
        after: 'I now have a tester who\'s always there, knows what I know better than I do, and becomes more personalized to my understanding over time!' },
      { before: 'Organizing my studying is hard. I find it hard to keep track of my formative assessments, homework, and flashcards in different places.',
        after: 'I can store, view, and test myself with these auto-generated quizzes on demand, all in one app!' },
    ],

  }

  render() {

    return (
      <div className="App">
        <div style={{display: 'flex', flex: 1, flexDirection: 'column'}}>
          <div class="hero__mask" style={{
            height: this.state.height * ( 1 + 2/3 * (1/0.975))}}></div>
          <div class="hero__overlay hero__overlay--gradient" style={{height: this.state.height * ( 1 + 2/3 * (1/0.975))}}></div>
          <div style={{display: 'flex', flexDirection: 'column', height: this.state.height * 2/21, width: this.state.width * 0.95, marginLeft: this.state.width * 0.025, marginRight: this.state.width * 0.025,}}>
            <Header about={this.state.about} style={{zIndex: 5,}}/>
          </div>
          <div style={{display: 'flex', backgroundColor: 'gray', flexDirection: 'column', height: (573/420 + 4/21) * this.state.height, width: this.state.width * 0.95, marginLeft: this.state.width * 0.025, marginRight: this.state.width * 0.025, marginBottom: this.state.height * 0.025}}>
            <Interface />
          </div>
        </div>
      </div>
    );
  }
}

export default Quizkly;
