import React from 'react';

const Title = () => {

  return (
    <div style={{flex: 4, display: 'flex', flexDirection: 'row', alignItems: 'stretch', backgroundColor: 'yellow'}}>
      <div style={{display: 'flex', flex: 1, flexDirection: 'column', justifyContent: 'center', backgroundColor: 'white'}}>
        <h5 style={{fontSize: '10/3vh'}}>Quizzes</h5>
      </div>
      <div style={{display: 'flex', flex: 4, flexDirection: 'column', justifyContent: 'center', backgroundColor: 'white'}}>
        <span style={{fontSize: '10vh'}}>Quizkly</span>
      </div>
      <div style={{display: 'flex', flex: 1, flexDirection: 'column', justifyContent: 'center', backgroundColor: 'white'}}>
        <h5 style={{fontSize: '10/3vh'}}>...quickly!</h5>
      </div>
    </div>
  );

}

export default Title;