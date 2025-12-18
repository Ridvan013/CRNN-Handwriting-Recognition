import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../App';

function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        const success = await login(username, password);
        if (success) {
            navigate('/');
        } else {
            setError('Invalid credentials. Please try again.');
        }
    };

    return (
        <div className="login-page" style={{ minHeight: '100vh' }}>
            <div className="background-glow"></div>

            <div className="login-container">
                <div className="login-card">
                    <div className="logo">
                        <i className="fa-solid fa-brain"></i>
                        <h1>Pro<span>OCR</span></h1>
                    </div>
                    <p className="subtitle">Teacher Access Portal</p>

                    {error && (
                        <div className="flash-messages">
                            <div className="alert">{error}</div>
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="login-form">
                        <div className="input-group">
                            <i className="fa-solid fa-user"></i>
                            <input
                                type="text"
                                name="username"
                                placeholder="Username"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                required
                            />
                        </div>
                        <div className="input-group">
                            <i className="fa-solid fa-lock"></i>
                            <input
                                type="password"
                                name="password"
                                placeholder="Password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                        </div>
                        <button type="submit" className="login-btn">
                            Login <i className="fa-solid fa-arrow-right"></i>
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}

export default Login;
