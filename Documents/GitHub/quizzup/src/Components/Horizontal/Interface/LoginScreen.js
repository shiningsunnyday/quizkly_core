import React from 'react';
import './Interface.css';
import { Link } from 'react-router-dom';
import CSRFToken from './CSRFToken.js';

const LoginScreen = (props) => {

  return (
    <div style={{display: 'flex', flexDirection: 'column', position: 'absolute', width: '30%', height: '70%', left: '35%', top: '15%', backgroundColor: 'green'}}>
      <form onSubmit={props.loginSubmit}>
        <CSRFToken />
        <label className="loginLabel">
          <h3>Username</h3>
          <input name="username" style={{position: 'relative', height: '50%', top: '50%'}} type="text" value={props.value} onChange={props.handleChangeUserName} />
        </label>
        <label className="loginLabel">
          <h3>Password</h3>
          <input name="password" type="text" value={props.value} onChange={props.handleChangePassword} />
        </label>
        <button type="submit" className="loginButton">Hi press me</button>
      </form>

    </div>
  );

}

export default LoginScreen;


// <label className="loginLabel">
//   <h3>Username</h3>
//   <input name="username" style={{position: 'relative', height: '50%', top: '50%'}} type="text" value={props.value} onChange={props.handleChangeUserName} />
// </label>
// <label className="loginLabel">
//   <h3>Password</h3>
//   <input name="password" type="text" value={props.value} onChange={props.handleChangePassword} />
// </label>
// <button className="loginButton" onClick={props.loginSubmit}><Link id="loginButton" to="/app"></Link></button>
