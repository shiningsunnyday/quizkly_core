import React from 'react';
import './Interface.css';
import Quiz from './Quiz.js';

const Quizzes = (props) => {
  console.log(props.documents, " are documents");
  return (
    <div style={styles.quizzes}>
      {props.documents.map((document, index) => {
        return <Quiz index={index} quiz={document.quiz} />
      })}
    </div>
  );
}

const styles = {
  quizzes: {
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: 'white',
  }
}

export default Quizzes;
