import { useState, useEffect } from 'react';
import { useAuth } from '../App';
import { Link } from 'react-router-dom';
import api from '../api';

function Profile() {
    const { user, logout, checkAuth } = useAuth();
    const [loading, setLoading] = useState(true);
    const [profileData, setProfileData] = useState(null);
    const [editMode, setEditMode] = useState(false);
    const [flashMessages, setFlashMessages] = useState([]);

    // Forms state
    const [profileForm, setProfileForm] = useState({ full_name: '', email: '' });
    const [passwordForm, setPasswordForm] = useState({ current_password: '', new_password: '', confirm_password: '' });

    useEffect(() => {
        fetchProfileData();
    }, []);

    const fetchProfileData = async () => {
        try {
            const response = await api.get('/profile');
            setProfileData(response.data);
            setProfileForm({
                full_name: response.data.user.full_name || '',
                email: response.data.user.email || ''
            });
            setLoading(false);
        } catch (error) {
            console.error('Failed to fetch profile', error);
        }
    };

    const addFlash = (type, message) => {
        const id = Date.now();
        setFlashMessages(prev => [...prev, { id, type, message }]);
        setTimeout(() => {
            setFlashMessages(prev => prev.filter(m => m.id !== id));
        }, 5000);
    };

    const handleProfileUpdate = async (e) => {
        e.preventDefault();
        try {
            const res = await api.post('/edit_profile', profileForm);
            if (res.data.success) {
                addFlash('success', 'Profile updated successfully!');
                checkAuth(); // Update global user state
                fetchProfileData(); // Update local data
                setEditMode(false);
            } else {
                addFlash('error', 'Failed to update profile.');
            }
        } catch (err) {
            addFlash('error', 'Failed to update profile.');
        }
    };

    const handlePasswordChange = async (e) => {
        e.preventDefault();
        if (passwordForm.new_password !== passwordForm.confirm_password) {
            addFlash('error', 'New passwords do not match!');
            return;
        }
        try {
            const res = await api.post('/change_password', passwordForm);
            if (res.data.success) {
                addFlash('success', 'Password changed successfully!');
                setPasswordForm({ current_password: '', new_password: '', confirm_password: '' });
            }
        } catch (err) {
            addFlash('error', err.response?.data?.error || 'Failed to change password.');
        }
    };

    if (loading || !profileData) {
        return (
            <div className="loading-section" style={{ height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <div className="loader"><div className="scanner"></div></div>
            </div>
        );
    }

    const { user: userData, predictions, total_predictions } = profileData;

    return (
        <div>
            <div className="background-glow"></div>
            <div className="container">
                <header>
                    <div className="header-content">
                        <div className="logo">
                            <i className="fa-solid fa-brain"></i>
                            <h1>Pro<span>OCR</span></h1>
                        </div>
                        <div className="user-menu">
                            <Link to="/" className="nav-link">
                                <i className="fa-solid fa-home"></i> Home
                            </Link>
                            <span className="user-info">
                                <i className="fa-solid fa-user-circle"></i> {user?.full_name || 'Teacher'}
                            </span>
                            <button onClick={logout} className="logout-btn" title="Logout">
                                <i className="fa-solid fa-right-from-bracket"></i> Logout
                            </button>
                        </div>
                    </div>
                    <p className="subtitle">User Profile & Settings</p>
                </header>

                {flashMessages.map(msg => (
                    <div key={msg.id} className="flash-container">
                        <div className={`alert alert-${msg.type}`}>
                            <i className={`fa-solid fa-${msg.type === 'success' ? 'check-circle' : 'exclamation-circle'}`}></i>
                            {msg.message}
                        </div>
                    </div>
                ))}

                <main className="profile-main">
                    <div className="profile-grid">
                        {/* User Info Card */}
                        <div className="profile-card">
                            <div className="card-header">
                                <h2><i className="fa-solid fa-user"></i> Profile Information</h2>
                                <button className="edit-btn" onClick={() => setEditMode(!editMode)}>
                                    {editMode ? <><i className="fa-solid fa-eye"></i> View</> : <><i className="fa-solid fa-pen-to-square"></i> Edit</>}
                                </button>
                            </div>

                            {!editMode ? (
                                <div className="profile-info">
                                    <div className="info-row">
                                        <span className="label">Username:</span>
                                        <span className="value">{userData.username}</span>
                                    </div>
                                    <div className="info-row">
                                        <span className="label">Full Name:</span>
                                        <span className="value">{userData.full_name || 'Not set'}</span>
                                    </div>
                                    <div className="info-row">
                                        <span className="label">Email:</span>
                                        <span className="value">{userData.email || 'Not set'}</span>
                                    </div>
                                    <div className="info-row">
                                        <span className="label">Role:</span>
                                        <span className="value role-badge">{userData.role || 'Teacher'}</span>
                                    </div>
                                    <div className="info-row">
                                        <span className="label">Member Since:</span>
                                        <span className="value">{userData.created_at?.slice(0, 10)}</span>
                                    </div>
                                    <div className="info-row">
                                        <span className="label">Last Login:</span>
                                        <span className="value">{userData.last_login ? userData.last_login.slice(0, 16) : 'Never'}</span>
                                    </div>
                                </div>
                            ) : (
                                <form onSubmit={handleProfileUpdate}>
                                    <div className="form-group">
                                        <label>Full Name</label>
                                        <input
                                            type="text"
                                            value={profileForm.full_name}
                                            onChange={e => setProfileForm({ ...profileForm, full_name: e.target.value })}
                                            required
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label>Email</label>
                                        <input
                                            type="email"
                                            value={profileForm.email}
                                            onChange={e => setProfileForm({ ...profileForm, email: e.target.value })}
                                            required
                                        />
                                    </div>
                                    <div className="form-actions">
                                        <button type="submit" className="save-btn">
                                            <i className="fa-solid fa-check"></i> Save Changes
                                        </button>
                                        <button type="button" className="cancel-btn" onClick={() => setEditMode(false)}>
                                            <i className="fa-solid fa-times"></i> Cancel
                                        </button>
                                    </div>
                                </form>
                            )}
                        </div>

                        {/* Change Password Card */}
                        <div className="profile-card">
                            <div className="card-header">
                                <h2><i className="fa-solid fa-lock"></i> Change Password</h2>
                            </div>
                            <form onSubmit={handlePasswordChange} className="password-form">
                                <div className="form-group">
                                    <label>Current Password</label>
                                    <input
                                        type="password"
                                        name="current_password"
                                        value={passwordForm.current_password}
                                        onChange={e => setPasswordForm({ ...passwordForm, current_password: e.target.value })}
                                        required
                                    />
                                </div>
                                <div className="form-group">
                                    <label>New Password</label>
                                    <input
                                        type="password"
                                        name="new_password"
                                        value={passwordForm.new_password}
                                        onChange={e => setPasswordForm({ ...passwordForm, new_password: e.target.value })}
                                        required
                                        minLength="6"
                                    />
                                </div>
                                <div className="form-group">
                                    <label>Confirm New Password</label>
                                    <input
                                        type="password"
                                        name="confirm_password"
                                        value={passwordForm.confirm_password}
                                        onChange={e => setPasswordForm({ ...passwordForm, confirm_password: e.target.value })}
                                        required
                                        minLength="6"
                                    />
                                </div>
                                <button type="submit" className="save-btn">
                                    <i className="fa-solid fa-key"></i> Update Password
                                </button>
                            </form>
                        </div>

                        {/* Stats Card */}
                        <div className="profile-card">
                            <div className="card-header">
                                <h2><i className="fa-solid fa-chart-simple"></i> Statistics</h2>
                            </div>
                            <div className="stats-grid">
                                <div className="stat-item">
                                    <div className="stat-icon">
                                        <i className="fa-solid fa-image"></i>
                                    </div>
                                    <div className="stat-info">
                                        <span className="stat-value">{total_predictions}</span>
                                        <span className="stat-label">Total Predictions</span>
                                    </div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-icon">
                                        <i className="fa-solid fa-clock"></i>
                                    </div>
                                    <div className="stat-info">
                                        <span className="stat-value">{predictions.length}</span>
                                        <span className="stat-label">Recent Activity</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* History Card */}
                        <div className="profile-card history-card">
                            <div className="card-header">
                                <h2><i className="fa-solid fa-history"></i> Recent Predictions</h2>
                            </div>
                            <div className="history-list">
                                {predictions && predictions.length > 0 ? (
                                    predictions.map((pred, i) => (
                                        <div key={i} className="history-item">
                                            <div className="history-header">
                                                <span className="history-image">
                                                    <i className="fa-solid fa-file-image"></i> {pred.image_name}
                                                </span>
                                                <span className="history-date">{pred.created_at?.slice(0, 16)}</span>
                                            </div>
                                            <div className="history-text">
                                                <div className="text-label">Corrected:</div>
                                                <div className="text-value corrected">{pred.corrected_text}</div>
                                            </div>
                                            {pred.raw_text !== pred.corrected_text && (
                                                <div className="history-text">
                                                    <div className="text-label">Raw:</div>
                                                    <div className="text-value raw">{pred.raw_text}</div>
                                                </div>
                                            )}
                                        </div>
                                    ))
                                ) : (
                                    <div className="empty-state">
                                        <i className="fa-solid fa-inbox"></i>
                                        <p>No predictions yet. Start by uploading an image!</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </main>

                <footer>
                    <p>Powered by CRNN + CRAFT + Trigram LM</p>
                </footer>
            </div>
        </div>
    );
}

export default Profile;
